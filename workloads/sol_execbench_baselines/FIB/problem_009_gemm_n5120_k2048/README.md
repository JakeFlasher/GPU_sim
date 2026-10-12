# problem_009_gemm_n5120_k2048

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | fp16 GEMM `C = A @ B.T` with A=[M, K], B=[N, K], C=[M, N]; N=5120 and K=2048 pinned, M varies. Captured from Qwen 3 30B A3B `attn.qkv_proj`. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_009_gemm_n5120_k2048.md`               |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/FlashInfer-Bench/009_gemm_n5120_k2048/`                       |
| Variant                     | `cutlass`                                                                                   |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

CUTLASS device-side fp16 GEMM with fp32 accumulator, mirroring the
landed problem_004 / problem_007 GEMM templates. A is RowMajor, B is
described as ColumnMajor (which IS the row-major `B = [N, K]` viewed
as `B^T = [K, N]` column-major with leading dim K), and C is
RowMajor — so the kernel emits `C = A @ B^T` natively without a
transpose copy.

- ElementA / ElementB / ElementC = `cutlass::half_t`;
  ElementAccumulator = `float`.
- TensorOp on Sm80 (forward-compatible with the local 3090 Ti sm_86
  host and with the Phase-2 B200 sm_100a target — Sm80 PTX is
  preserved by both).
- ThreadblockShape `<128, 128, 32>` — the standard Ampere fp16 tile.
  Tiles N=5120 into 40 N-tiles per row-stripe (5120 / 128 = 40).
- WarpShape `<64, 64, 32>` keeps four warps per CTA; InstructionShape
  `<16, 8, 16>` is the m16n8k16 fp16 mma on Sm80.
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
  HBM3e (~8 TB/s) reshapes the K-reduction streaming bandwidth
  envelope. The short K=2048 is largely L2-resident across hosts,
  so the bandwidth gap matters less here than on the K=14336 sibling
  (problem_008); the dominant cost is the m16n8k16 MMA throughput.
- DSMEM and cluster-cooperative B-tile staging (sm_100a-only) are
  out of scope at Sm80 portability level. A Phase-2 follow-up could
  switch to a Sm100 CUTLASS schedule (CollectiveBuilder /
  Hopper-style mainloop) that uses TMA for global → shared staging
  and cluster-wide cooperative store of the C-tile.
- L2 carveout / partition: B is [5120, 2048] = 20 MiB fp16 — fits
  inside B200's larger L2 but spills the 3090 Ti's 6 MB L2 by ~3x.
  The L2-hit pattern on the B-stream is therefore substantially
  worse on the 3090 Ti than on B200.
- TCGen05 / TMEM divergence: Sm80 mma.m16n8k16 instructions are used;
  the B200 TCGen05 / TMEM path is NOT exercised here. A future tuning
  pass to a Sm100-specific schedule (Hopper-style mainloop with TMA
  + tcgen05.mma) would lift that.
