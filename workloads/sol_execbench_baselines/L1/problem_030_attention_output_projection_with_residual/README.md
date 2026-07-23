# problem_030_attention_output_projection_with_residual

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | BF16 fused attention output projection with inline residual addition. `output = (attn_output @ o_proj_weight.T) + residual`. Captured from AI21-Jamba-Reasoning-3B. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_030_attention_output_projection_with_residual.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/030_attention_output_projection_with_residual/`            |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

CUDA C++ baseline that fuses the attention output projection with
the residual-add into a single CUTLASS Sm80 TensorOp BF16 GEMM
via the `D = α · A · B + β · C` epilogue (with α = 1, β = 1).

- attn_output `[B, S, hidden_size=2560]` bf16 is flattened to
  `[B*S, 2560]` and passed as the GEMM A matrix.
- o_proj_weight `[2560, 2560]` bf16 row-major is described as
  ColumnMajor with leading dim 2560 — that view IS the
  transpose, so the GEMM computes
  `attn_output @ o_proj_weight.T` natively without a separate
  transpose copy.
- residual `[B, S, 2560]` bf16 is flattened and passed as the
  GEMM C source. The CUTLASS epilogue reads each output-tile
  element from C, scales by β = 1, and adds it to the
  accumulator before the BF16 store.
- output `[B, S, 2560]` bf16 is the GEMM D destination.
- ThreadblockShape 128 × 128 × 32, WarpShape 64 × 64 × 32,
  InstructionShape 16 × 8 × 16, 3-stage pipeline. fp32
  accumulator with the standard Ampere `mma.m16n8k16.bf16`
  instructions.

`main.cpp` pins `hidden_size == 2560` via `TORCH_CHECK`. The
residual-add is folded into the same kernel pass that produces
the projection output, eliminating the intermediate write the
PyTorch reference path emits.

This baseline expands the cohort's GEMM regime beyond the
residual-free `D = α · A · B` path used by problem_004 through
problem_011. The C-stream sectors flow through the epilogue
alongside the GEMM output, doubling the per-CTA non-K
bandwidth demand.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-
checked and natively-correctness-validated on. Final SOL scoring
is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060):
  the C-stream (residual) doubles the off-K bandwidth versus
  a residual-free GEMM. B200's HBM3e (~8 TB/s) absorbs this
  more easily than the 3090 Ti's GDDR6X (~1 TB/s); local
  wall-time is a weak proxy for B200 SOL Score on this slug.
- DSMEM and cluster-cooperative C-tile staging (sm_100a-only)
  are out of scope at Sm80 portability level. A Phase-2
  follow-up could switch to a Sm100 schedule that streams the
  residual via TMA + cluster-cooperative load to amortize the
  C-stream across a CTA cluster.
- L2 carveout / partition: with hidden_size = 2560, the
  o_proj_weight tile is 2560 × 2560 × 2 = 12.5 MiB bf16, which
  overflows the 3090 Ti's 6 MB L2 but fits in B200's larger
  L2 partition. The L2-hit signal on cohort labels reflects
  the per-tile reuse the threadblock swizzle gives, not whole-
  weight residency.
- TCGen05 / TMEM divergence: Sm80 mma.m16n8k16 instructions are
  used; the B200 TCGen05 / TMEM path is NOT exercised here.
