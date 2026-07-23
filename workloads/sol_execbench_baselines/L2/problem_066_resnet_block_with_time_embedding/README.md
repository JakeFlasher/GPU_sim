# problem_066_resnet_block_with_time_embedding

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | ResNet block with diffusion-style time embedding (conv + temb fold + residual).             |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_066_resnet_block_with_time_embedding.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L2/066_resnet_block_with_time_embedding/`                     |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Compile-checked-sm_120`                                                                    |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                      |
| Local speedup vs reference  | <diagnostic only on sm_86>                                                                  |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | sm_86 the calibration workstream: static + cupti_activity ok; sanitizer / nsys / nvbit_metrics skipped to cap R12 wall-clock. |

## Approach

Hand-port of the diffusion ResNet block reference. The block
takes a hidden state and a time-embedding vector and applies:
conv2d → groupnorm → SiLU → time-embedding bias fold → conv2d →
groupnorm → SiLU → residual add. FP16 inputs and outputs; FP32
accumulators in the conv and norm reductions.

## sm_86 the calibration workstream result (R12)

All 16 sol-execbench workloads PASSED. Sidecar at
`scripts/baselines_harness/.harness_traces/problem_066_resnet_block_with_time_embedding__cuda_cpp__sm86/full_correctness_v1.json`;
SHA-256 cross-stamped in bundle manifest + ledger.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) compiles + runs this baseline correctly;
Phase-2 profiling runs on B200 (sm_100a). Cross-arch divergence
axes:

- HBM3e vs GDDR6X / GDDR7 bandwidth: the ResNet block streams
  intermediate activations between conv stages, so memory
  bandwidth matters; B200's HBM3e changes the ceiling.
- DSMEM and cluster-cooperative reduction (sm_100a-only) — could
  amortise the GroupNorm reduction across CTAs on B200, but not
  used here.
- L2 carveout: B200's larger L2 may keep activation tiles
  resident across the two conv stages.
- TCGen05 / TMEM divergence: 5th-gen tensor cores accelerate the
  conv matmuls on B200; only relevant once TCGen05 paths land.
