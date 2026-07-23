# problem_034_flux_multi_axis_rope_frequency_computation

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | fp32 multi-axis RoPE frequency computation for FLUX 3D position encoding (time, height, width). Generates per-axis cos/sin frequency tables and concatenates them. Two distinct destination-passing-style outputs from a single launcher. Captured from black-forest-labs/FLUX.1-Fill-dev. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_034_flux_multi_axis_rope_frequency_computation.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/034_flux_multi_axis_rope_frequency_computation/`           |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

CUDA C++ baseline that computes the `[seq_len, total_dim=128]`
cos and sin frequency tables for FLUX 3D position encoding by
running 1D RoPE independently per axis
(`axes_dim = [16, 56, 56]`) and concatenating along the last
dimension.

Algorithm per row `s` and output column `d`:
- Resolve `(axis, pair_idx, half_dim)` from `d` via branched
  arithmetic over the three axis ranges
  `d ∈ [0, 16) → axis 0, half_dim = 8`
  `d ∈ [16, 72) → axis 1, half_dim = 28`
  `d ∈ [72, 128) → axis 2, half_dim = 28`
- Load `axis_pos = ids[s, axis]` (a single fp32 broadcast
  across the 16 / 56 / 56 threads of that axis range).
- Compute `freq_band = 1 / powf(theta, pair_idx / half_dim)`
  on the fly.
- Compute `angle = axis_pos * freq_band`.
- Write `freqs_cos[s, d] = cosf(angle)` and
  `freqs_sin[s, d] = sinf(angle)`.

The repeat-interleave-by-2 over the half_dim cos/sin pairs is
absorbed by the `d >> 1` derivation of `pair_idx`: adjacent
`d` and `d + 1` share the same `pair_idx`, hence the same
`(axis_pos, freq_band, angle)` and identical output values
in the cos/sin tables.

Kernel topology:
- One CTA per `seq_len` position. Grid: `(seq_len,)`.
- `BLOCK_THREADS = total_dim = 128` (one thread per output
  `d` element). Each kernel invocation populates BOTH outputs
  in parallel — distinct from the "two sequential launches"
  multi-output-DPS pattern used elsewhere in the cohort.

`main.cpp` pins `n_axes == 3` and `total_dim == 128` via
`TORCH_CHECK`; validates 2D ids + 2D freqs_cos / freqs_sin
shape consistency, fp32 dtype, CUDA + contiguous. Receives
`theta` as `double` (matching the reference `theta: float`
signature) and downcasts to `float` at the launch boundary.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-
checked and natively-correctness-validated on. Final SOL scoring
is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060):
  this kernel is compute-bound (almost zero read traffic from
  `ids`; heavy `powf` / `sinf` / `cosf` evaluation). Memory
  bandwidth differences barely affect throughput. B200's
  larger SM count and updated SFU pipeline should give the
  larger gain.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are
  out of scope; no reduction here.
- L2 carveout / partition: the `ids` reads are tiny (3 fp32
  per row CTA = 12 B) and entirely L1-resident; output writes
  are pure stream-stores that bypass L1 / L2 hit-rate
  semantics for the read-only sector counters used by the
  cohort-B calibration carveout.
- TCGen05 / TMEM divergence: 5th-gen tensor cores and TMEM
  register surface only exist on sm_100a; this kernel uses
  only SFU ops (powf, sinf, cosf).
