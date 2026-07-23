# problem_006_gemm_n2048_k4096

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | fp16 GEMM `C = A @ B.T` at N=2048, K=4096; M varies per workload.                           |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_006_gemm_n2048_k4096.md`               |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/FlashInfer-Bench/006_gemm_n2048_k4096/`                       |
| Variant                     | `cutlass`                                                                                   |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Same CUTLASS GEMM template as the batch-001 GEMM landings, retargeted
to the (N=2048, K=4096) shape:

- `cutlass::gemm::device::Gemm<…>` with TensorOp on `Sm80`.
- Threadblock shape `128x128x32`, warp shape `64x64x32`,
  instruction shape `16x8x16`. 3-stage pipeline. The larger
  threadblock (vs the 64x128x32 used for the smaller-N variants)
  amortises tile setup over the wider N=2048 axis.
- `RowMajor A × ColumnMajor B = RowMajor C` (B is passed as row-
  major `[N, K]` from PyTorch, which is bit-identical to
  ColumnMajor `[K, N]` with leading dim K).
- Epilogue: `LinearCombination` with alpha=1, beta=0. fp32
  accumulator → fp16 output.

## Architecture-coupling caveats

`Ran-local-sm_86` reflects local correctness on the RTX 3090 Ti host
only. The B200 (sm_100a) path is unvalidated until Phase 2.

- HBM3e vs GDDR6X / GDDR7 bandwidth: at N=2048 / K=4096, the GEMM
  is compute-bound for large M but bandwidth-bound for small M;
  B200's HBM3e has roughly 20× the bandwidth of the 3090 Ti's
  GDDR6X and ~15× the bandwidth of the future cross-arch sm_120
  RTX 5060's GDDR7.
- Sm80 TensorOp path; no sm_100a-only features (TCGen05 / TMEM /
  TMA / cluster / DSMEM). The Phase-2 retest may reach a different
  occupancy on Blackwell.
- L2 carveout: 6 MiB L2 on 3090 Ti is much smaller than B200's L2;
  expect a higher L2 hit rate when re-profiled on B200.
- 5th-gen tensor cores on sm_100a use TCGen05 instruction encodings
  and TMEM register surface; the Sm80 mma.sync path here is the
  generic tensor-core lowering compatible with both arches.

## Calibration cohort

`predictor_friendly`. Tiled GEMM access pattern with high temporal
locality on the operand reuse.
