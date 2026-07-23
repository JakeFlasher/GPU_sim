# problem_087_embedding_with_initial_layernorm_backward

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | Embedding lookup + initial LayerNorm backward (FP16 inputs / FP32 gradient).                |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_087_embedding_with_initial_layernorm_backward.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/087_embedding_with_initial_layernorm_backward/`            |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Compile-checked-sm_120`                                                                    |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                      |
| Local speedup vs reference  | <diagnostic only on sm_86>                                                                  |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | sm_86 the calibration workstream: static + cupti_activity ok; sanitizer / nsys / nvbit_metrics skipped to cap R12 wall-clock. |

## Approach

Hand-port of the embedding + initial LayerNorm backward reference.
The forward path performed embedding lookup + LayerNorm; this
baseline implements the backward pass that accumulates
`grad_embedding` from `grad_output` through the LayerNorm Jacobian
(mean / variance reductions) and scatters via index_add. FP16
inputs, FP32 accumulators for the LayerNorm-Jacobian terms, FP32
gradient output.

## sm_86 the calibration workstream result (R12)

All 16 sol-execbench workloads PASSED. Sidecar at
`scripts/baselines_harness/.harness_traces/problem_087_embedding_with_initial_layernorm_backward__cuda_cpp__sm86/full_correctness_v1.json`;
SHA-256 cross-stamped in bundle manifest + ledger.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) compiles + runs this baseline correctly;
Phase-2 profiling runs on B200 (sm_100a). Cross-arch divergence
axes:

- HBM3e vs GDDR6X / GDDR7 bandwidth: backward kernels are
  bandwidth-bound on the gradient accumulator scatter.
- DSMEM and cluster-cooperative reduction (sm_100a-only): could
  amortise the LayerNorm-Jacobian reductions across CTAs on
  B200; not used here.
- L2 carveout: B200's larger L2 may keep the LayerNorm parameter
  weights resident across the forward-saved-stats reload.
- TCGen05 / TMEM divergence: not used; this kernel has no matmul.
