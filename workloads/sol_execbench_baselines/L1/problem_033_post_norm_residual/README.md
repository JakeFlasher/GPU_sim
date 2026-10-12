# problem_033_post_norm_residual

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | BF16 post-norm residual connection: `output = residual + RMSNorm(sublayer_output, weight, eps)`. RMSNorm runs in fp32 (bf16→fp32→bf16); residual add applied in bf16 after the downcast. Captured from Olmo3. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_033_post_norm_residual.md`             |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/033_post_norm_residual/`                                   |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

CUDA C++ baseline that mirrors the existing
`problem_002_fused_add_rmsnorm_h4096` template (per-row CTA,
BF16x2 vector `__ldg`, fp32 sum-of-squares with warp shuffle +
cross-warp shared-memory reduction) with the *inversion* that
makes this slug distinct:

- The existing `fused_add_rmsnorm` family mixes the residual
  into the input BEFORE normalization (`RMSNorm(x +
  residual)`).
- This slug applies RMSNorm to the sublayer output FIRST and
  adds the residual OUTSIDE the normalization
  (`output = residual + RMSNorm(sublayer_output)`).
- The bf16 → fp32 → bf16 cast happens INSIDE RMSNorm; the
  residual add operates on bf16 + bf16 (matching the
  reference's explicit `normalized.to(input_dtype)` step
  before the `residual + normalized` add).

Layout:
- `sublayer_output`, `residual`, `output`: `[B, S, H=4096]`
  bf16 row-major.
- `weight`: `[H]` bf16. Reused across rows.
- `eps`: fp32 scalar passed at launch time (the reference
  signature uses `eps: float`; the DPS entry takes
  `double eps` and downcasts to `float`).

Kernel topology:
- One CTA per row over `B * S` rows.
- BLOCK_THREADS = 256, stride-loop over
  `HIDDEN_SIZE_VEC = 2048` BF16x2 pairs.
- Pass 1: fp32 sum-of-squares of `sublayer_output`. Warp
  shuffle + cross-warp shared-memory tree reduction.
- Pass 2: re-load `sublayer_output`, `residual`, and
  `weight`; compute the normalized value in fp32, downcast to
  bf16, add the residual via `__hadd2`, store.

`main.cpp` pins `hidden_size == 4096` via `TORCH_CHECK`.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-
checked and natively-correctness-validated on. Final SOL scoring
is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5090):
  this kernel is bandwidth-bound (read 3× `H` bf16 per row —
  sublayer_output, residual, weight — write 1× `H` bf16).
  B200's HBM3e (~8 TB/s) lifts the per-row throughput ceiling
  ~8× over GDDR6X.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are
  out of scope at Sm80 portability level; this kernel uses
  per-CTA shared memory and warp shuffles only.
- L2 carveout / partition: at modest batch sizes the per-row
  data fits in L2; for very large `B*S` the working set
  crosses the 3090 Ti's 6 MB L2 boundary.
- TCGen05 / TMEM divergence: 5th-gen tensor cores and TMEM
  register surface only exist on sm_100a; this kernel does
  not use tensor cores (the math is pointwise bf16 / fp32
  multiplies and adds with one reduction), so the TCGen05 /
  TMEM split is not exercised here.
