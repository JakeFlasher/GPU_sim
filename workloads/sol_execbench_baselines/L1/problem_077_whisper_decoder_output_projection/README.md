# problem_077_whisper_decoder_output_projection

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | fp16 GEMM `logits = hidden_states @ weight.T` with hidden_states=[B, S, d_model], weight=[vocab_size, d_model], logits=[B, S, vocab_size]; d_model=1280 and vocab_size=51866 pinned, M=B*S varies. Captured from openai/whisper-large-v3 decoder output projection. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_077_whisper_decoder_output_projection.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/077_whisper_decoder_output_projection/`                    |
| Variant                     | `cutlass`                                                                                   |
| Evidence                    | `Compile-checked-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                             |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

CUTLASS device-side fp16 GEMM with fp32 accumulator, mirroring the
landed problem_004 / problem_007 / problem_008 / problem_009 /
problem_010 / problem_011 GEMM-cohort templates. The 3D
`[B, S, *]` input and output tensors are flattened to 2D
`[B*S, *]` via `flatten(0, 1)` in the entry-point launcher; the
GEMM itself is purely 2D with the same RowMajor / ColumnMajor
encoding the cohort already uses:

- A (hidden_states_2d): [M=B*S, K=1280] row-major → RowMajor lda=K.
- B (weight): [N=51866, K=1280] row-major → ColumnMajor ldb=K
  (the `B^T` col-major [K, N] view).
- C (logits_2d): [M, N=51866] row-major → RowMajor ldc=N.

Template instantiation:

- ElementA / ElementB / ElementC = `cutlass::half_t`;
  ElementAccumulator = `float`.
- TensorOp on Sm80 (forward-compatible with the local 3090 Ti sm_86
  host and the Phase-2 B200 sm_100a target — Sm80 PTX is preserved
  by both).
- ThreadblockShape `<128, 128, 32>`, WarpShape `<64, 64, 32>`,
  InstructionShape `<16, 8, 16>` — the standard Ampere fp16
  mma.m16n8k16 tile.
- NumStages = 3, swizzle = `GemmIdentityThreadblockSwizzle`.
- Epilogue: `LinearCombination<half_t, 2, float, float>` —
  **AlignmentC = 2**, not the default 8.

### Why AlignmentC = 2

`vocab_size = 51866` is NOT 8-element aligned: 51866 mod 8 = 2.
The default `EpilogueOp` would derive `kAlignmentC =
128 / sizeof_bits<half_t> = 8` (8-element fp16 vector stores), and
the 8-wide store at the tail tile would either trip kernel-init
validation or stomp 6 bytes past the output buffer. 51866 mod 2 = 0,
so `AlignmentC = 2` (2-element fp16 vector stores = 32-bit aligned)
satisfies the contract while keeping a vectorised store path.

AlignmentA and AlignmentB stay at the default 8 because
`K = d_model = 1280` is 8-aligned (1280 mod 8 = 0).

### Runtime-check audit

Every const axis from `definition.json` is enforced on every input
and output tensor the kernel indexes with that fixed assumption:

| Tensor          | Runtime check                                              |
|-----------------|------------------------------------------------------------|
| `hidden_states` | `is_cuda`, `kHalf`, `is_contiguous`, `dim == 3`, `size(2) == d_model = 1280` |
| `weight`        | `is_cuda`, `kHalf`, `is_contiguous`, `dim == 2`, `size(0) == vocab_size = 51866`, `size(1) == d_model = 1280` |
| `logits`        | `is_cuda`, `kHalf`, `is_contiguous`, `dim == 3`, `size(0) == hidden_states.size(0)`, `size(1) == hidden_states.size(1)`, `size(2) == vocab_size = 51866` |

Destination-passing-style entry: `logits` is pre-allocated by the
eval driver. The launcher uses the same pointer for both `ref_C`
(source) and `ref_D` (dest) of the GEMM Arguments, with `beta = 0`
so any existing contents are ignored.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-
checked and natively-correctness-validated on. Final SOL scoring
is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060): B200's
  HBM3e reshapes the K-reduction streaming envelope. The B-stream
  here is `[51866, 1280]` fp16 = ~130 MiB — far outside both the
  3090 Ti's 6 MB L2 and B200's L2 partition; the kernel is
  effectively cold-streaming on the B-stream on both hosts.
- DSMEM and cluster-cooperative B-tile staging (sm_100a-only) are
  out of scope at Sm80 portability level. A Phase-2 follow-up could
  switch to a Sm100 CUTLASS schedule (CollectiveBuilder /
  Hopper-style mainloop) that uses TMA for global → shared staging.
- TCGen05 / TMEM divergence: Sm80 mma.m16n8k16 is used; the B200
  TCGen05 / TMEM path is NOT exercised here.
- L2 carveout / partition: workload-dependent. Small-M workloads
  (B=1, S=1) stream a tiny A tile through L1, but the B-tile is
  pure HBM streaming regardless of M.
