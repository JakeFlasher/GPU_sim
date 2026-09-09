# problem_040_conv2d_residual_block

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | Conv2D residual block (Conv2D + BN-style scale/shift + residual add + ReLU).                |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_040_conv2d_residual_block.md`          |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/040_conv2d_residual_block/`                                |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Compile-checked-sm_120`                                                                    |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                      |
| Local speedup vs reference  | <diagnostic only on sm_86>                                                                  |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | sm_86 the calibration workstream: static + cupti_activity ok; sanitizer / nsys / nvbit_metrics skipped to cap R12 wall-clock. |

## Approach

Hand-port of the Conv2D residual block reference. Standard
3×3 conv2d followed by per-channel scale + shift, residual add of
the input, and a ReLU. FP16 inputs and outputs; the conv kernel
delegates the matmul-style reduction so the matmul precision
matches the PyTorch reference (per
BL-20260528-delegate-matmul-to-at-matmul-for-reference-hgemm-
precision-parity if applicable).

## sm_86 the calibration workstream result (R12)

All 16 sol-execbench workloads PASSED. Sidecar at
`tools/sol_baselines_harness/.harness_traces/problem_040_conv2d_residual_block__cuda_cpp__sm86/full_correctness_v1.json`;
SHA-256 cross-stamped in bundle manifest + ledger.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) compiles + runs this baseline correctly;
Phase-2 profiling runs on B200 (sm_100a). Cross-arch divergence
axes:

- HBM3e vs GDDR6X / GDDR7 bandwidth: conv2d is bandwidth-sensitive
  for the residual / activation path; B200's HBM3e changes the
  ceiling.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are out
  of scope.
- L2 carveout: B200's larger L2 may absorb activation tiles the
  local consumer GPU's smaller L2 cannot.
- TCGen05 / TMEM divergence: 5th-gen tensor cores accelerate
  conv2d matmuls on B200; only relevant once TCGen05 paths are
  enabled.
