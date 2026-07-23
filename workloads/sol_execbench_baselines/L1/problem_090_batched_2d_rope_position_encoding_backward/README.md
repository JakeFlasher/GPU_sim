# problem_090_batched_2d_rope_position_encoding_backward

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | BF16/fp32 backward pass for batched 2D RoPE position encoding. Computes `grad_idx_theta = -grad_cos.f32 * sin(idx_theta) + grad_sin.f32 * cos(idx_theta)` per element. Single fp32 DPS output. Pure per-element kernel — no reductions, no broadcasts, no head-axis collapse. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_090_batched_2d_rope_position_encoding_backward.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/090_batched_2d_rope_position_encoding_backward/`            |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

CUDA C++ baseline implementing the per-element backward pass
for batched 2D RoPE position encoding.

Algorithm per element:
- `sin_t = sinf(idx_theta[b, s, d])`
- `cos_t = cosf(idx_theta[b, s, d])`
- `g_cos = bf16_to_f32(grad_cos[b, s, d])`
- `g_sin = bf16_to_f32(grad_sin[b, s, d])`
- `grad_idx_theta[b, s, d] = -g_cos * sin_t + g_sin * cos_t`

No reductions, no broadcasts, no shared memory. Every output
element is produced from exactly three input elements at the
same `(b, s, d)` position.

Kernel topology:
- One CTA per `(b, s)` row of `head_dim` elements.
  Grid: `(seq_len, batch_size)`.
- `BLOCK_THREADS = 256`. Each thread stride-loops over the
  row's `head_dim` elements via a `for d = tid; d < head_dim;
  d += kBlockThreads` loop. `head_dim` is workload-variable
  but bounded by typical RoPE head dimensions.
- Each thread loads `idx_theta[idx]` (fp32), computes
  `sinf(idx_t)` and `cosf(idx_t)` via the SFU, loads
  `grad_cos[idx]` and `grad_sin[idx]` as BF16 and casts to
  fp32, then writes the per-element output.

`main.cpp` validates 3D shape consistency across all four
tensors, BF16 dtype for `grad_cos` / `grad_sin`, fp32 dtype
for `idx_theta` / `grad_idx_theta`, CUDA + contiguous. DPS
entry signature
`run(grad_cos, grad_sin, idx_theta, grad_idx_theta)`.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was
compile-checked and natively-correctness-validated on.
Final SOL scoring is on B200 (sm_100a). Differences worth
noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060):
  this kernel reads each input element once and writes
  each output element once. Three loads (BF16 +
  BF16 + fp32) + one fp32 write per element. B200's HBM3e
  (~8 TB/s) doubles the bandwidth-bound throughput.
- DSMEM and cluster-cooperative reduction (sm_100a-only)
  are out of scope; no reduction here.
- L2 carveout / partition: each per-(b, s) row tile is
  `head_dim * (2 + 2 + 4 + 4) = head_dim * 12 B` of
  traffic — small enough to fit easily in L1 for typical
  head dims (64-128).
- TCGen05 / TMEM divergence: 5th-gen tensor cores and
  TMEM register surface only exist on sm_100a; this kernel
  uses no tensor cores.
- SFU pipeline: each output element requires one
  `sinf` + one `cosf` SFU evaluation. SFU throughput
  often dominates arithmetic intensity for this style of
  per-element trig kernel.
