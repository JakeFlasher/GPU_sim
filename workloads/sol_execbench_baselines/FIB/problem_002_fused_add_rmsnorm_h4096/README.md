# problem_002_fused_add_rmsnorm_h4096

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | Fused residual-add + RMSNorm at hidden_size=4096 for Llama-3.1-8B; EPS=1e-5.                |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_002_fused_add_rmsnorm_h4096.md`        |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/FlashInfer-Bench/002_fused_add_rmsnorm_h4096/`                |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                          |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                           |
| Local speedup vs reference  | 3.10 – 13.56× on sm_120 (diagnostic only)                                                   |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Same proven fused-add-rmsnorm pattern as pilot #1 (`problem_001_fused_add_rmsnorm_h2048`),
scaled to the larger Llama-3.1-8B hidden size:

- One CTA per row (`[batch, hidden_size=4096]`).
- 512 threads per block, each handling 4 BF16x2 pairs (the row has
  2048 BF16x2 pairs total).
- Two passes: sum-of-squares of `(hidden_states + residual)` in FP32
  with a warp-shuffle all-reduce then a single-warp cross-warp
  reduction; output rescale using
  `inv_rms = rsqrtf(row_sum / hidden_size + EPS)` and store
  `(h + r) * inv_rms * weight` in BF16.
- EPS pinned at the value the problem definition specifies (1e-5).
  Wider FP32 accumulation per CLAUDE.md Rule 8.
- BF16 inputs / outputs, FP32 intermediates. No CUDA streams created
  by us, no `data_ptr`-keyed caching, no monkey-patching — kernel
  launches on `at::cuda::getCurrentCUDAStream()`.

Local sm_120 result: 14/14 PASSED, `speedup_factor` 3.10 - 13.56×.
The reference's PyTorch path issues a residual add kernel, a
`pow(2).mean(...)` kernel, a `+ EPS` + `rsqrt` kernel, and a final
multiply-by-weight kernel; this single fused kernel collapses the chain
at no precision cost.

## Architecture-coupling caveats

The local 5060 (sm_120) compiles and smoke-runs this baseline; final
SOL scoring is on B200 (sm_100a). Differences worth noting for this
pattern:

- HBM3e vs GDDR7 bandwidth: this kernel is bandwidth-bound (read two
  `[B, 4096]` BF16 tensors + 1D weight, write `[B, 4096]` BF16).
  B200's HBM3e has roughly 20× the bandwidth of the 5060's GDDR7, so
  local wall-time is a weak proxy for B200 SOL Score — the kernel
  will be even more bandwidth-dominant there.
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
