# problem_056_mamba_ssm_dt_projection_softplus_clamp_backward

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | BF16 backward pass for Mamba SSM dt-projection with softplus and clamp. Two DPS outputs: per-element `grad_dt[B, S, H]` and per-channel `grad_dt_bias[H]` (fp32-summed-then-bf16-cast reduction). |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_056_mamba_ssm_dt_projection_softplus_clamp_backward.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/056_mamba_ssm_dt_projection_softplus_clamp_backward/`       |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

CUDA C++ baseline that implements the backward pass for
Mamba SSM dt-projection with softplus and clamp in a single
per-head CTA kernel.

Algorithm per `(row = b*S + s, h)`:
- `clamp_pass = (time_step_min < dt_activated < time_step_max) ? 1.0f : 0.0f` (gradient passes only if strictly within bounds; boundary values are non-differentiable).
- `softplus_grad = 1 / (1 + exp(-dt_with_bias.f32)) = sigmoid(dt_with_bias)`.
- `grad = grad_output.f32 * clamp_pass * softplus_grad` (all fp32).
- `grad_dt[row, h] = bf16(grad)` (per-element BF16 store).
- `grad_dt_bias[h] = bf16(Σ_(b, s) grad)` (fp32 sum then BF16 cast).

`num_heads = 40` is pinned by the problem definition;
`batch_size` and `seq_len` are workload-variable.

### Kernel topology

- **Grid**: `(num_heads,)`. One CTA per `h`. Each CTA
  processes ALL `(b, s)` rows for a single head.
- **BLOCK_THREADS = 256** (8 warps). Stride-loop over
  `num_rows = B*S`.
- Each thread maintains a fp32 `partial_sum` accumulator
  while writing `grad_dt` per-element.
- Warp-shuffle + 8-way cross-warp tree reduces
  `partial_sum`; thread 0 casts to BF16 and writes
  `grad_dt_bias[h]`.

### Read-access pattern

Each warp's 32 contiguous threads read 32 contiguous
`(b, s)` rows at offset `h` within each row. The stride
between adjacent threads' loads is
`num_heads = 40 elements = 80 bytes` — a strided
non-coalesced read by design. This is the slug's natural
memory access shape; the cohort-B probe mirrors it
exactly.

### Numerical contract

The reference computes `grad.sum(dim=(0, 1)).to(torch.bfloat16)`
— summing in fp32 BEFORE the BF16 cast. My kernel matches
this by accumulating in fp32 throughout and casting to BF16
only when writing the final scalar. Casting to BF16 before
summing would lose precision and fail correctness on
larger workloads.

`main.cpp` pins `num_heads == 40` via `TORCH_CHECK`;
validates 3D shape consistency, 1D `grad_dt_bias[H]`, BF16
dtype, CUDA + contiguous. `time_step_min` and
`time_step_max` arrive as `double` (matching the
reference `float` signature) and are downcast to `float`
at the launch boundary. DPS entry signature
`run(grad_output, dt_with_bias, dt_activated,
time_step_min, time_step_max, grad_dt, grad_dt_bias)` —
5 inputs and 2 outputs.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was
compile-checked and natively-correctness-validated on.
Final SOL scoring is on B200 (sm_100a).

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5090):
  this kernel is bandwidth-bound on the BF16 inputs and
  outputs; the strided read pattern further widens the
  effective bandwidth requirement.
- DSMEM and cluster-cooperative reduction (sm_100a-only)
  are out of scope; this kernel uses only CTA-local
  shared memory for the 8-way cross-warp tree.
- L2 carveout / partition: the three input streams each
  have `B*S*40*2 B = B*S*80 B` of read traffic per CTA;
  the per-CTA read amount scales with B*S. Workloads
  with `B*S * 80 B` larger than the per-SM 192 KiB L1
  spill to L2.
- TCGen05 / TMEM divergence: this kernel uses no tensor
  cores.
