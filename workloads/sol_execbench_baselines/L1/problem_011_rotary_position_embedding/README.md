# problem_011_rotary_position_embedding

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | RoPE cos/sin embedding from `position_ids[batch, seq]` + `inv_freq[64]` + scalar `attention_scaling`; output `cos_sin[batch, seq, 128, 2]` bf16. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_011_rotary_position_embedding.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/011_rotary_position_embedding/`                            |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

One CTA per `(batch, seq)` pair; `half_head_dim = 64` threads per
block, one per first-half lane:

- inv_freq[64] is staged into shared memory once per CTA (256 B; the
  same llama3-scaled table is reused for every position in the block).
- Each thread reads `pos = position_ids[batch, seq]`, computes
  `freq = inv_freq[tid] * pos` in FP32, then `__sincosf(freq, &s, &c)`
  to get a single (cos, sin) pair scaled by `attention_scaling`.
- The PyTorch reference builds `emb = cat(freqs, freqs)` along
  head_dim, so the cos/sin pair at lane `tid` is bit-identical to the
  pair at lane `tid + 64`. Each thread fans its pair into BOTH output
  slots — one scalar write per `(cos | sin, head_dim_slot)`.
- Output layout `[batch, seq, head_dim, 2]` is contiguous; writes are
  scalar bf16 stores. A future tuning pass could vectorise the four
  writes into a single 64-bit `ushort4` once correctness is in.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-
checked and natively-correctness-validated on. Final SOL scoring is
on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060): the kernel
  is bandwidth-light per CTA (the inv_freq stage is 256 B and the
  output is 512 B per row), so achievable bandwidth differs by
  factors across hosts but the pattern is identical.
- DSMEM and cluster cooperative load (sm_100a-only) are unused; the
  shared-memory stage is per-CTA, not per-cluster.
- L2 carveout / cache residency: inv_freq[64] is trivially L2-resident
  on every host. The output write stream is strictly write-only and
  is not L2-read.
- TCGen05 / TMEM divergence: this kernel does not use tensor cores at
  all (the math is `sincos` + scalar multiply), so the TCGen05 / TMEM
  split is not exercised on B200 vs sm_86 / sm_120.
