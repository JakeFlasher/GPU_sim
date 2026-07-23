# problem_024_rmsnorm_h2048

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | Plain Root Mean Square Normalization at hidden_size=2048 (no residual add); from Qwen3-30B-A3B, EPS=1e-6. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_024_rmsnorm_h2048.md`                  |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/FlashInfer-Bench/024_rmsnorm_h2048/`                          |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                          |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                           |
| Local speedup vs reference  | 2.66 – 12.68× on sm_120 (diagnostic only)                                                   |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

This is the plain (non-residual) RMSNorm variant at the smaller hidden
size. The kernel re-uses the standard RMSNorm structure used in the
cohort's other RMSNorm hand-ports, with the residual add path
omitted:

- One CTA per row (`[batch, hidden_size=2048]`).
- 256 threads per block, each handling 4 BF16x2 pairs (the row has 1024
  BF16x2 pairs total) — smaller block than pilot #1's 1024-thread
  per-row configuration because the smaller hidden size doesn't need it
  and the smaller block lets more CTAs live per SM.
- Two passes: sum-of-squares in FP32 with a warp-shuffle all-reduce
  then a single-warp cross-warp reduction; output rescale using
  `inv_rms = rsqrtf(row_sum / hidden_size + EPS)`.
- EPS is pinned at the value the problem definition specifies (1e-6).
  Wider FP32 accumulation is preserved per CLAUDE.md Rule 8.
- BF16 inputs / outputs, FP32 intermediates. No CUDA streams created
  by us, no `data_ptr`-keyed caching, no monkey-patching — kernel
  launches on `at::cuda::getCurrentCUDAStream()`.

Local sm_120 result: 7/7 PASSED, `speedup_factor` 2.66 - 12.68×. The
reference's PyTorch path (`hs.pow(2).mean(...) + EPS` then a separate
multiply by `weight`) issues several small kernels per row; this single
fused kernel collapses the chain.

## Architecture-coupling caveats

The local 5060 (sm_120) compiles and smoke-runs this baseline; final
SOL scoring is on B200 (sm_100a). Differences worth noting for this
pattern:

- HBM3e vs GDDR7 bandwidth: this kernel is bandwidth-bound (read
  `[B, 2048]` BF16, write `[B, 2048]` BF16). B200's HBM3e has roughly
  20× the bandwidth of the 5060's GDDR7, so local wall-time is a weak
  proxy for B200 SOL Score — the kernel will be even more
  bandwidth-dominant there.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are out of
  scope for sm_120; this kernel uses per-CTA shared memory and warp
  shuffles only.
- L2 carveout: at modest batch sizes the entire row stays in L1 / L2;
  for very large batches B200's larger L2 may absorb a working set
  the 5060's smaller L2 cannot.
- TCGen05 / TMEM divergence: 5th-gen tensor cores and TMEM register
  surface only exist on sm_100a; this kernel does not use tensor cores
  (the math is pointwise BF16/FP32 multiplies and adds), so the
  TCGen05 / TMEM split is not exercised here.
