# problem_012_fused_cos_sin_embedding_generation

| Field                       | Value                                                                                                |
|-----------------------------|------------------------------------------------------------------------------------------------------|
| Problem                     | Fused cos/sin RoPE embedding generation; FP32 freqs in, BF16 cos/sin out; head_dim=128              |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_012_fused_cos_sin_embedding_generation.md`     |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/012_fused_cos_sin_embedding_generation/`                            |
| Variant                     | `cuda_cpp`                                                                                           |
| Evidence                    | `Profiled-local-sm_120`                                                                              |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                               |
| Local speedup vs reference  | <pending local 3090 Ti run; diagnostic only>                                                         |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                                     |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                                    |

## Approach

One CTA per (batch, seq) row, 128 threads per CTA (one thread per
output position in the head_dim axis). The reference's
`emb = cat(freqs, freqs)` replication is folded into the index math:
for lane `d` in `[0, head_dim)`, the input index is
`d % half_head_dim`, so lanes `[0, 64)` and `[64, 128)` read the
same `freqs[..., d % 64]` value. Each thread computes both
`cos(freqs)` and `sin(freqs)` via `__sincosf`, applies the FP32
`attention_scaling` multiplier, and casts to BF16 via round-to-
nearest-even.

`head_dim = 128` and `half_head_dim = 64` are baked into the
definition; `batch_size` and `seq_len` are variable axes. Launch is
on `at::cuda::getCurrentCUDAStream()` per CLAUDE.md Rule 7. Inputs
are validated for `is_cuda` / dtype / contiguity at the
PyBind11 boundary; outputs are written in destination-passing style.

## Architecture-coupling caveats

The local 3090 Ti (sm_86 / GA102) compiles and runs this baseline;
final SOL scoring is on B200 (sm_100a / GB100). Differences worth
noting for this pattern:

- HBM3e vs GDDR6X bandwidth: this kernel is bandwidth-bound (small
  per-output compute, ~12 bytes IO per output element). The 3090
  Ti's GDDR6X ceiling makes local wall-time a weak proxy for the
  B200 SOL Score.
- DSMEM and cluster-cooperative reduction (sm_100a-only): not used
  here. The per-row work is independent so DSMEM is not exercised
  by this pattern; B200 cluster support adds no new behaviour.
- L2 cache: B200's larger L2 absorbs no significant working-set
  beyond the freqs input (which is read once per row); the L2
  carveout difference is not a load-bearing factor for this kernel.
- TCGen05 / TMEM divergence: this kernel does not touch tensor
  cores at all (no MACs; cos/sin are SFU intrinsics). TCGen05 and
  TMEM are not exercised; the kernel is portable across sm_86 / 89
  / 90 / 100 / 120 without architecture-dependent codegen.
- GDDR7 vs GDDR6X: the consumer Blackwell 5060 also uses GDDR7;
  the 3090 Ti uses GDDR6X. The pattern is bandwidth-bound so this
  affects local wall-clock but not correctness.

## Counter-free artifacts

Counter-free analytics bundle path:
`tools/sol_baselines_harness/.harness_traces/problem_012_fused_cos_sin_embedding_generation__cuda_cpp/`
(produced by the M4 surfaces orchestrator fanout; status pending
the local 3090 Ti pipeline run that is part of M4).
