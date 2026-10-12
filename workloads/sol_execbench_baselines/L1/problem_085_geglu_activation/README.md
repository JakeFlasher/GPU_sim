# problem_085_geglu_activation

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | FP32 GEGLU (Gated GLU with GELU) activation. Splits input on last axis into (gate, linear), returns `GELU_tanh(gate) * linear`. Captured from SDXL refiner. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_085_geglu_activation.md`               |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/085_geglu_activation/`                                     |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

CUDA C++ device-side fp32 GEGLU activation — element-wise pointwise
kernel with float4 vectorized loads/stores along the contiguous last
dim.

- Input `x` is `[B, S, 2 * inner_dim]` fp32 row-major; output is
  `[B, S, inner_dim]` fp32 row-major. `inner_dim = 5120` pinned by
  the problem definition.
- One thread per output float4 (4 consecutive cols within the same
  row). Each thread reads one float4 from the gate half (offset 0
  within the row) and one float4 from the linear half (offset
  `inner_dim` within the row), then writes one float4 of output.
- GELU uses the tanh approximation:
  `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
  `tanhf` matches the reference's tolerance contract.
- `inner_dim=5120` is divisible by 4 and ATen-allocated tensors
  are 16-byte aligned, so float4 alignment holds for both halves
  in every row — no tail kernel needed.
- BLOCK_THREADS=256; grid sizes to `n_rows * (inner_dim/4)` float4
  output positions.

`main.cpp` pins `inner_dim == 5120` via `TORCH_CHECK` so a workload
with a drifted inner_dim fails closed rather than silently
degrading.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-checked
and natively-correctness-validated on. Final SOL scoring is on B200
(sm_100a). Differences worth noting for this pattern:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5090): this kernel
  is bandwidth-bound (read 2x `inner_dim` fp32, write 1x `inner_dim`
  fp32 per row). B200's HBM3e (~8 TB/s) reshapes the per-row streaming
  envelope; local wall-time is a weak proxy for B200 SOL Score.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are out of
  scope here — GEGLU is pure element-wise with no shared-memory
  coordination.
- L2 carveout / partition: at modest batch sizes the per-row gate /
  linear halves fit in L2; for very large batches B200's larger L2
  partition absorbs a working set the 3090 Ti's 6 MB L2 cannot.
- TCGen05 / TMEM divergence: 5th-gen tensor cores and TMEM register
  surface only exist on sm_100a; this kernel does not use tensor
  cores, so the TCGen05 / TMEM split is not exercised here.
