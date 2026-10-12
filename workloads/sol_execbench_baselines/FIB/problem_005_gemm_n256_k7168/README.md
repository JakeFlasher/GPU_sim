# problem_005_gemm_n256_k7168

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | fp16 GEMM `C = A @ B.T` at N=256, K=7168; M varies per workload.                            |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_005_gemm_n256_k7168.md`                |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/FlashInfer-Bench/005_gemm_n256_k7168/`                        |
| Variant                     | `cutlass`                                                                                   |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Same CUTLASS GEMM template as the batch-001 GEMM landings
(`problem_004_gemm_n128_k2048`, `problem_007_gemm_n4096_k4096`),
re-targeted to the (N=256, K=7168) shape:

- `cutlass::gemm::device::Gemm<…>` with TensorOp on `Sm80` —
  same code path compiles for sm_86 local + sm_100a B200.
- Threadblock shape `64x128x32`, warp shape `32x64x32`,
  instruction shape `16x8x16`. 3-stage pipeline.
- `RowMajor A × ColumnMajor B = RowMajor C`. PyTorch passes
  `B [N, K]` in row-major, which is identical to `B.T [K, N]` in
  column-major with leading dim K — so the CUTLASS arg uses
  ColumnMajor for B without a physical transpose.
- Epilogue: `LinearCombination` with alpha=1, beta=0. fp32
  accumulator → fp16 output.
- Destination-passing style: the caller allocates `C [M, N]` and
  CUTLASS writes in-place via the same pointer.

## Architecture-coupling caveats

`Ran-local-sm_86` reflects local correctness on the RTX 3090 Ti
host only. The B200 (sm_100a) path is unvalidated until Phase 2.

- HBM3e vs GDDR6X / GDDR7 bandwidth: K=7168 means each A-tile
  loads ~14 KiB of fp16 per row-tile and each B-tile loads ~14
  KiB per column-tile; the inner GEMM is bandwidth-bound at low
  M. B200's HBM3e has roughly 20× the bandwidth of the 3090 Ti's
  GDDR6X and ~15× the bandwidth of the future cross-arch sm_120
  RTX 5090's GDDR7.
- Sm80 TensorOp path; no sm_100a-only features (TCGen05 / TMEM /
  TMA / cluster / DSMEM). The Phase-2 retest is expected to
  compile cleanly but may reach a different occupancy /
  register profile.
- L2 carveout: 6 MiB L2 on 3090 Ti vs much larger L2 on B200;
  expect a higher L2 hit rate when re-profiled on B200.
- 5th-gen tensor cores on sm_100a use TCGen05 instruction
  encodings and TMEM register surface; the Sm80 path here lowers
  to mma.sync (16x8x16 fp16 fp32-accumulate), so the cross-arch
  microbench split is not exercised by this kernel.

## Calibration cohort

`predictor_friendly`. The CUTLASS GEMM has a regular tiled access
pattern that the cache predictor handles well.
