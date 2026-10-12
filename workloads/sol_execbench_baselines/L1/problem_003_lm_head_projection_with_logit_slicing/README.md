# problem_003_lm_head_projection_with_logit_slicing

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | bf16 GEMM `logits = hidden_states[:, -logits_to_keep:, :] @ weight.T` with hidden_states=[B, S, hidden_size], weight=[vocab_size, hidden_size], logits=[B, logits_to_keep, vocab_size]; hidden_size=2048 and vocab_size=102400 pinned. All canonical workloads pin logits_to_keep == seq_len. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_003_lm_head_projection_with_logit_slicing.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/003_lm_head_projection_with_logit_slicing/`                |
| Variant                     | `cutlass`                                                                                   |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

CUTLASS device-side bf16 GEMM with fp32 accumulator, mirroring the
landed problem_004 / problem_007 / problem_008 / problem_009 /
problem_010 / problem_011 GEMM-cohort templates. The 3D
`[B, S, *]` input and `[B, logits_to_keep, *]` output tensors are
flattened to 2D `[B*S, *]` and `[B*logits_to_keep, *]` via
`flatten(0, 1)`; the GEMM itself is 2D with the same
RowMajor / ColumnMajor encoding the cohort already uses:

- A (hidden_states_2d): [M=B*S, K=2048] row-major → RowMajor lda=K.
- B (weight): [N=102400, K=2048] row-major → ColumnMajor ldb=K
  (the `B^T` col-major [K, N] view).
- C (logits_2d): [M, N=102400] row-major → RowMajor ldc=N.

Template instantiation:

- ElementA / ElementB / ElementC = `cutlass::bfloat16_t`;
  ElementAccumulator = `float`.
- TensorOp on Sm80.
- ThreadblockShape `<128, 128, 32>`, WarpShape `<64, 64, 32>`,
  InstructionShape `<16, 8, 16>` — Ampere bf16 mma.m16n8k16.
- NumStages = 3, `GemmIdentityThreadblockSwizzle`.
- Epilogue: default `LinearCombination<bfloat16_t,
  128 / sizeof_bits<bfloat16_t> = 8, float, float>`.
  **AlignmentC = 8** (102400 mod 8 = 0).

### Logit-slicing handling

The reference computes
`logits = hidden_states[:, -logits_to_keep:, :] @ weight.T`. Every
one of the 16 canonical workloads has `logits_to_keep == seq_len`
(verified directly from `workload.jsonl`), so the slice is a no-op.

This baseline fast-paths the no-slice case and explicitly enforces
the canonical-workload invariant via
`TORCH_CHECK(logits.size(1) == hidden_states.size(1))`. Workloads
with `logits_to_keep < seq_len` would need either a per-batch input
pointer offset GEMM or a `cutlass::gemm::device::GemmBatched`
schedule; those code paths are explicitly out of scope until the
canonical workload set exercises them.

### Runtime-check audit

Every const axis from `definition.json` is enforced on every input
and output tensor the kernel indexes with that fixed assumption:

| Tensor          | Runtime check                                              |
|-----------------|------------------------------------------------------------|
| `hidden_states` | `is_cuda`, `kBFloat16`, `is_contiguous`, `dim == 3`, `size(2) == hidden_size = 2048` |
| `weight`        | `is_cuda`, `kBFloat16`, `is_contiguous`, `dim == 2`, `size(0) == vocab_size = 102400`, `size(1) == hidden_size = 2048` |
| `logits`        | `is_cuda`, `kBFloat16`, `is_contiguous`, `dim == 3`, `size(0) == hidden_states.size(0)`, `size(1) == hidden_states.size(1)` (canonical: `logits_to_keep == seq_len`), `size(2) == vocab_size = 102400` |

Destination-passing-style entry: `logits` is pre-allocated by the
eval driver. The launcher uses the same pointer for both `ref_C`
(source) and `ref_D` (dest) of the GEMM Arguments with `beta = 0`.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-
checked and natively-correctness-validated on. Final SOL scoring is
on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5090): B200's
  HBM3e reshapes the K-reduction streaming envelope. The B-stream
  here is `[102400, 2048]` bf16 = ~400 MiB — far outside both the
  3090 Ti's 6 MB L2 and B200's L2 partition; effectively cold-
  streaming on the B-stream on both hosts.
- DSMEM and cluster-cooperative B-tile staging (sm_100a-only) are
  out of scope at Sm80 portability level.
- TCGen05 / TMEM divergence: Sm80 mma.m16n8k16 is used; the B200
  TCGen05 / TMEM bf16 path is NOT exercised here.
- L2 carveout / partition: workload-dependent. The A-stream
  `[M, 2048]` is small for small-M workloads; the B-stream is
  pure HBM streaming regardless.
