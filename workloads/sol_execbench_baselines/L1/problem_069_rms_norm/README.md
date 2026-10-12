# problem_069_rms_norm

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | Fused residual-add + RMSNorm, hidden_size=8192, 3D [batch, seq, hidden] inputs              |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_069_rms_norm.md`                       |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/069_rms_norm/`                                             |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                          |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                           |
| Local speedup vs reference  | 2.51 – 9.61× on sm_120 (diagnostic only)                                                    |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

One CTA per (batch, seq) row (rows flattened to `batch_size * seq_len`).
Threads cooperatively load BF16x2 pairs of `hidden_states` and `residual`,
sum `(h + r)^2` per element in FP32, and reduce across the block via warp-
shuffle plus shared-memory inter-warp reduction. `inv_rms = rsqrtf(mean_sq
+ eps)` then drives the rewrite `(h + r) * inv_rms * weight` in FP32 with a
final cast to BF16 on store. `hidden_size=8192` is baked in as a
`constexpr`; the kernel uses 1024 threads / 4 BF16x2 elements per thread.
Launch is on `at::cuda::getCurrentCUDAStream()` per CLAUDE.md Rule 7.

### Local sm_120 harness result (16 / 16 workloads PASSED)

| Workload | Status | speedup_factor |
|---|---|---|
| 0  | PASSED | 6.57 |
| 1  | PASSED | 2.51 |
| 2  | PASSED | 8.63 |
| 3  | PASSED | 2.58 |
| 4  | PASSED | 9.14 |
| 5  | PASSED | 9.35 |
| 6  | PASSED | 3.34 |
| 7  | PASSED | 8.89 |
| 8  | PASSED | 4.95 |
| 9  | PASSED | 8.86 |
| 10 | PASSED | 4.95 |
| 11 | PASSED | 8.51 |
| 12 | PASSED | 8.48 |
| 13 | PASSED | 2.53 |
| 14 | PASSED | 9.61 |
| 15 | PASSED | 4.91 |

Hardware: NVIDIA GeForce RTX 5090 GPU (sm_120). Evaluation ran
through the native `sol-execbench` CLI (no Docker); the cache /
manifest identify this run by `native_env_hash` (see
`tools/sol_baselines_harness/native_env.py`).

## Architecture-coupling caveats

The local 5090 (sm_120) compiles and smoke-runs this baseline; final SOL
scoring is on B200 (sm_100a). Differences worth noting for this pattern:

- HBM3e vs GDDR7 bandwidth: this kernel is heavily memory-bound (each
  row reads 2 × 16 KB inputs + 16 KB weight, writes 16 KB output); the
  5090's GDDR7 ceiling makes local wall-time a weak proxy for the
  B200 SOL Score. `seq_len` × `batch_size` row count drives total
  bandwidth pressure linearly.
- DSMEM and cluster-cooperative reduction (sm_100a-only): the current
  kernel uses one CTA per row, so DSMEM is not exercised. A cluster
  variant could pool the per-row reduction across CTAs at very large
  hidden_size; for `hidden_size=8192` the single-CTA reduction is
  sufficient.
- L2 carveout: B200's larger L2 absorbs the `weight` tensor (16 KB)
  across all rows, eliminating repeated HBM traffic; the 5090's
  smaller L2 may not retain it across the row count of the larger
  workloads, surfacing as bandwidth pressure that B200 would not see.
- TCGen05 / TMEM divergence: this kernel does not touch 5th-generation
  tensor cores or the TMEM register surface; the rmsnorm path is
  pure SIMT FP32 / BF16 arithmetic with no matrix-multiply step, so
  it is not exposed to sm_100a-only TCGen05 / TMEM codegen
  differences.
