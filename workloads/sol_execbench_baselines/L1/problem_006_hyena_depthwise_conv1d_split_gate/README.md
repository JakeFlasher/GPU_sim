# problem_006_hyena_depthwise_conv1d_split_gate

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | Hyena depthwise 1D conv + split-gate (FP16 inputs/outputs).                                 |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_006_hyena_depthwise_conv1d_split_gate.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/006_hyena_depthwise_conv1d_split_gate/`                    |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Compile-checked-sm_120`                                                                    |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                      |
| Local speedup vs reference  | 1.7 – 2.5× on sm_86 (diagnostic only)                                                       |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | sm_86 the calibration workstream: cupti_activity / sanitizer / nsys / static ok, nvbit_metrics failed (parser bug on `sector_efficiency_global_load`). |

## Approach

Hand-port of the FlashInfer hyena depthwise conv1d + split-gate
reference. Per-channel 1D depthwise convolution followed by an
elementwise sigmoid gate split. FP16 inputs and outputs with FP32
accumulators for the conv reduction.

## sm_86 the calibration workstream result (R12)

All 16 sol-execbench workloads PASSED on this 3090 Ti host. The
full 16-row correctness sidecar is tracked at
`tools/sol_baselines_harness/.harness_traces/problem_006_hyena_depthwise_conv1d_split_gate__cuda_cpp__sm86/full_correctness_v1.json`;
its SHA-256 is cross-stamped in both the bundle manifest and
`tools/sol_baselines_harness/calibration/cache_prediction/sm86_pilot_correctness_v1.toml`.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) compiles + runs this baseline correctly;
Phase-2 counter-free profiling and SOL Score submission run on
B200 (sm_100a). Cross-arch divergence axes for this pattern:

- HBM3e vs GDDR6X / GDDR7 bandwidth — depthwise conv has a low
  arithmetic intensity, so B200's HBM3e ceiling differs sharply
  from local GDDR.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are out
  of scope for sm_86 / sm_120 builds; the kernel uses a single
  CTA per per-channel slice.
- L2 carveout: B200's larger L2 may absorb tensors the local
  consumer GPU's smaller L2 cannot.
- TCGen05 / TMEM divergence: not used; depthwise conv has no
  matmul step.
