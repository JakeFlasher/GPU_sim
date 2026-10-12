# problem_007_gemm_n4096_k4096

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | fp16 GEMM `C = A @ B.T` with A=[M, K], B=[N, K], C=[M, N]; N=4096 and K=4096 pinned, M varies. Captured from Llama 3.1 8B `attn.o_proj`. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_007_gemm_n4096_k4096.md`               |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/FlashInfer-Bench/007_gemm_n4096_k4096/`                       |
| Variant                     | `cutlass`                                                                                   |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

CUTLASS device-side fp16 GEMM with fp32 accumulator. Same layout
encoding as problem_004: A is RowMajor, B is described as ColumnMajor
(which IS the row-major `B = [N, K]` viewed as `B^T = [K, N]` column-
major with leading dim K), and C is RowMajor — so the kernel emits
`C = A @ B^T` natively without a transpose copy.

- ElementA / ElementB / ElementC = `cutlass::half_t`;
  ElementAccumulator = `float`.
- TensorOp on Sm80 (forward-compatible with the local 3090 Ti sm_86
  host and with the Phase-2 B200 sm_100a target — Sm80 PTX is
  preserved by both).
- ThreadblockShape `<128, 128, 32>` — the standard Ampere fp16 tile
  for large GEMMs. Tiles N=4096 into 32 N-tiles per row-stripe.
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
  envelope relative to the 3090 Ti / 5090 hosts. Large-M cases here
  are bandwidth-bound on the 3090 Ti; the same M on B200 will be
  compute-bound earlier.
- DSMEM and cluster-cooperative B-tile staging (sm_100a-only) are
  out of scope at Sm80 portability level. A Phase-2 follow-up could
  switch to a Sm100 CUTLASS schedule (CollectiveBuilder /
  Hopper-style mainloop) that uses TMA for global → shared staging
  and cluster-wide cooperative store of the C-tile.
- L2 carveout / partition: B is [4096, 4096] = 32 MiB fp16 — outside
  the 3090 Ti's 6 MB L2 but well inside B200's larger partition,
  so the L2-hit pattern on the B-stream is materially different
  across hosts.
- TCGen05 / TMEM divergence: Sm80 mma.m16n8k16 instructions are used;
  the B200 TCGen05 / TMEM path is NOT exercised here. A future tuning
  pass to a Sm100-specific schedule (Hopper-style mainloop with TMA
  + tcgen05.mma) would lift that.
