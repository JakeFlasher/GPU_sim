# problem_004_gemm_n128_k2048

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | fp16 GEMM `C = A @ B.T` with A=[M, K], B=[N, K], C=[M, N]; N=128 and K=2048 pinned, M varies (1 … 16294). |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_004_gemm_n128_k2048.md`                |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/FlashInfer-Bench/004_gemm_n128_k2048/`                        |
| Variant                     | `cutlass`                                                                                   |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

CUTLASS device-side fp16 GEMM with fp32 accumulator. The row-major
input layout for `B = [N, K]` is exactly column-major `B^T = [K, N]`
with leading dim `K`, so the kernel describes A as RowMajor / B as
ColumnMajor / C as RowMajor and emits a `C = A @ B^T` GEMM directly
without a transpose copy:

- ElementA / ElementB / ElementC = `cutlass::half_t`;
  ElementAccumulator = `float`.
- TensorOp on Sm80 (forward-compatible with the local 3090 Ti sm_86
  host and with the Phase-2 B200 sm_100a target — Sm80 PTX is
  preserved by both).
- ThreadblockShape `<64, 128, 32>` lines the N-tile up exactly with
  the pinned `N = 128`, so every threadblock covers a full row-stripe
  of C. The M-tile of 64 keeps the small-M cases (M = 1, 2, 4, …)
  from over-padding.
- WarpShape `<32, 64, 32>`, InstructionShape `<16, 8, 16>` — standard
  Ampere fp16 m16n8k16 mma.
- NumStages = 3, LinearCombination epilogue (alpha = 1, beta = 0).

Destination-passing-style entry: C is pre-allocated by the eval
driver and the launcher writes into the same pointer for both
`ref_C` (source) and `ref_D` (dest) of the GEMM, with beta = 0 so
the existing contents do not contribute.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-
checked and natively-correctness-validated on. Final SOL scoring is
on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5090): B200's
  HBM3e bandwidth (~8 TB/s) eclipses the 3090 Ti's GDDR6X
  (~1 TB/s) and the future 5090's GDDR7 (~0.5 TB/s) — large-M
  workloads will reshape into bandwidth- vs compute-bound differently
  on each host.
- DSMEM and cluster cooperative B-tile staging (sm_100a-only) are
  out of scope at Sm80 portability level; the per-CTA shared-memory
  staging is the only memory hierarchy in play.
- L2 carveout / partition: B is [128, 2048] = 512 KiB fp16, well
  inside the 3090 Ti's 6 MB L2 partition. On B200 the L2 cache /
  partition is much larger; the K-reduction loop is L2-cache-friendly
  on every host but the access pattern moves up the hierarchy on
  Blackwell.
- TCGen05 / TMEM divergence: Sm80 mma.m16n8k16 instructions are used;
  the B200 TCGen05 / TMEM path is NOT exercised here. A future tuning
  pass to a Sm100-specific schedule would lift that.
