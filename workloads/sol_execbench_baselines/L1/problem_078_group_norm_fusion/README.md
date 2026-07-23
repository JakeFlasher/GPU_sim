# problem_078_group_norm_fusion

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | fp32 GroupNorm for Stable Diffusion UNet: normalizes features across `(channels_per_group, H, W)` within each of `num_groups = 32` channel groups; applies per-channel weight/bias affine. Single DPS output. Captured from runwayml/stable-diffusion-v1-5. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_078_group_norm_fusion.md`              |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/078_group_norm_fusion/`                                    |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

CUDA C++ baseline that implements GroupNorm with two-pass
spatial reduction over each `(b, g)` group tile, followed by a
per-channel affine transform.

Algorithm per group `(b, g)`:
- `N` = `(C / num_groups) * H * W` elements in this group's
  tile.
- `mean` = `sum_{c_in_group, h, w} x[b, g*Cg+c_in_group, h, w] / N`
- `var`  = `sum_{c_in_group, h, w} x[...]^2 / N - mean^2`
- `inv_stddev` = `rsqrt(var + eps)`
- For each `(c_in_group, h, w)`:
  - `c = g * (C/num_groups) + c_in_group`
  - `output[b, c, h, w] = (x[b, c, h, w] - mean) * inv_stddev * weight[c] + bias[c]`

`num_groups = 32` is pinned by the problem definition;
`num_channels` is workload-variable but always a multiple of 32
(320, 640, 1280 in the workload sweep, giving channels_per_group
= 10, 20, 40).

Kernel topology:
- One CTA per `(b, g)` pair. Grid:
  `(num_groups=32, batch_size)`.
- `BLOCK_THREADS = 256` (8 warps). Each thread iterates over
  the group's `N = (C/G) * H * W` elements via a stride-loop,
  accumulating `partial_sum` and `partial_sum_sq` in fp32.
- Warp-shuffle reduction within each warp; the 8 warp leaders
  then cross-warp-reduce via shared memory. Thread 0 computes
  `mean`, `var = sum_sq/N - mean^2`, `inv_stddev = rsqrt(var
  + eps)`, and broadcasts both through shared floats.
- Pass 2 stride-loops again over the same N elements, looks up
  `weight[c]` / `bias[c]` for each element's global channel
  `c = g*(C/G) + c_in_group`, and writes
  `(x - mean) * inv_stddev * weight[c] + bias[c]` to output.

`main.cpp` pins `num_channels % 32 == 0` via `TORCH_CHECK`;
validates 4D x/output + 1D weight/bias shape consistency,
fp32 dtype, CUDA + contiguous. Receives `eps` as `double`
(matching the reference `eps: float` signature) and downcasts
to `float` at the launch boundary.

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
  (warp-shuffle + 8-way cross-warp tree) — would benefit from
  a cluster-cooperative all-reduce on sm_100a, but the per-CTA
  approach used here is portable.
- L2 carveout / partition: the per-channel weight + bias tables
  are tiny (`num_channels × 4 B` × 2 = 2.5/5/10 KiB across the
  three workload channel counts). They fit comfortably in L1.
  The input tensor reuse pattern depends on the group's tile
  size: small (`H × W = 8×8 = 64` elements per channel) groups
  fit in L1 across passes; large (`H × W = 64×64 = 4096`)
  groups force a re-read from L2 / HBM on pass 2.
- TCGen05 / TMEM divergence: 5th-gen tensor cores and TMEM
  register surface only exist on sm_100a; this kernel does
  not use tensor cores — pure element-wise scan + reduce.
