# problem_088_rotary_position_embedding_application

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | fp32 Rotary Position Embedding application: rotates query and key tensors by precomputed cos/sin tables. Two distinct destination-passing-style outputs. Captured from HunyuanImage-2.1. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_088_rotary_position_embedding_application.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/088_rotary_position_embedding_application/`                |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

CUDA C++ baseline that applies RoPE rotation to query and key
tensors via two distinct DPS outputs (the cohort's first
multi-output baseline).

Algorithm per element:
- For each `(b, h, s, d)`, compute:
  - `rotate_half[..., d]` = `-x[..., d+64]` if `d < 64`,
                            `+x[..., d-64]` if `d >= 64`
  - `output[..., d]` = `x[..., d] * cos[s, d] + rotate_half[..., d] * sin[s, d]`

The same kernel runs twice from the DPS entry — once for
`(query, query_rotated)` and once for `(key, key_rotated)`.
`num_heads` and `num_kv_heads` are workload-variable; only
`head_dim = 128` is pinned.

Kernel topology:
- One CTA per `(b, h, s)` row of `head_dim = 128` elements.
- `BLOCK_THREADS = 128` (one thread per element). Each thread
  reads its own input element, the rotate-half partner element,
  and the `cos[s, d]` + `sin[s, d]` table entries; multiplies
  and stores.
- Grid layout: `(seq_len, num_heads, batch_size)` so consecutive
  CTAs share the same `s` (and therefore the same row of `cos`
  and `sin`), giving strong inter-CTA reuse on the cos/sin
  loads.

`main.cpp` pins `head_dim == 128` via `TORCH_CHECK` and
validates that all four output tensors share the matching shape
contract.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-
checked and natively-correctness-validated on. Final SOL scoring
is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060):
  this kernel is bandwidth-bound (read x once + partner read,
  write x_rotated once, plus the small cos/sin table reads).
  B200's HBM3e (~8 TB/s) lifts the per-row throughput
  significantly.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are
  out of scope; this kernel uses no shared memory and no
  warp-level reductions.
- L2 carveout / partition: the cos/sin tables (`[S, 128]` fp32
  = `S * 512` bytes) fit in L2 easily for typical S (≤ 4096
  → 2 MiB); the inter-CTA reuse pattern drives strong L2 hit
  rates on cos/sin loads.
- TCGen05 / TMEM divergence: 5th-gen tensor cores and TMEM
  register surface only exist on sm_100a; this kernel does
  not use tensor cores, so the TCGen05 / TMEM split is not
  exercised here.
