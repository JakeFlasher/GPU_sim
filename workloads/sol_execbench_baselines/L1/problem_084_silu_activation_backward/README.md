# problem_084_silu_activation_backward

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | FP32 SiLU (Swish) backward. `grad_input = grad_output * sigmoid_x * (1 + x * (1 - sigmoid_x))`. Captured from SDXL base. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_084_silu_activation_backward.md`       |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/084_silu_activation_backward/`                             |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

CUDA C++ device-side fp32 SiLU backward — element-wise pointwise
kernel with float4 vectorized loads/stores along the flat 1D shape.
Same template structure as `problem_025_video_latent_gelu_activation`
(GELU forward) — both are 1D element-wise activations with a vec4
fast path and a scalar tail.

- Inputs `grad_output`, `x`, `sigmoid_x` are all flat
  `[num_elements]` fp32 contiguous; `grad_input` same shape (DPS).
- Each vec4 thread reads one float4 from each of the three inputs
  and writes one float4 to the output. Per-element math is
  `grad_input = grad_output * sigmoid_x * (1 + x * (1 - sigmoid_x))`
  — a direct transcription of the reference formula.
- Vec4 fast path covers the largest aligned prefix; the scalar tail
  kernel handles any 1-3-element remainder (the workload sweeps a
  range of `num_elements` values, not all multiples of 4).
- The reference caches `sigmoid_x` from the forward pass; the kernel
  consumes it directly rather than recomputing the sigmoid, matching
  the reference's chain-rule decomposition exactly.
- BLOCK_THREADS=256.

`main.cpp` validates that all four tensors are 1D and share `numel`
via `TORCH_CHECK`.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-checked
and natively-correctness-validated on. Final SOL scoring is on B200
(sm_100a). Differences worth noting for this pattern:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5090): this kernel
  is bandwidth-bound (read 3x N fp32, write 1x N fp32). B200's HBM3e
  (~8 TB/s) lifts the throughput ceiling significantly; local
  wall-time is a weak proxy for B200 SOL Score.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are out of
  scope here — SiLU backward is pure element-wise with no shared-
  memory coordination.
- L2 carveout / partition: at small `num_elements` the inputs stay
  in L2; for large `num_elements` the kernel becomes pure HBM
  streaming.
- TCGen05 / TMEM divergence: 5th-gen tensor cores and TMEM register
  surface only exist on sm_100a; this kernel does not use tensor
  cores, so the TCGen05 / TMEM split is not exercised here.
