# problem_025_rmsnorm_h4096

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | Plain Root Mean Square Normalization at hidden_size=4096 (no residual add); from Llama-3.1-8B, EPS=1e-5. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_025_rmsnorm_h4096.md`                  |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/FlashInfer-Bench/025_rmsnorm_h4096/`                          |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

CUDA C++ device-side plain RMSNorm (no residual add) — same template as
the landed problem_021 / problem_022 / problem_024 RMSNorm baselines,
specialised to hidden_size=4096. The kernel uses one CTA per row and
two `__ldg` passes (sum-of-squares first, then output multiply):

- One CTA per row (`[batch_size, hidden_size=4096]`).
- BLOCK_THREADS=256, with each thread handling 8 BF16x2 pairs via a
  stride loop over `HIDDEN_SIZE_VEC=2048` (4096 BF16 / 2). Same loop
  topology as problem_024 (h=2048) — wider per-thread span because
  hidden size doubled.
- Two passes: sum-of-squares in FP32 with a warp-shuffle all-reduce
  then a single-warp cross-warp reduction; output rescale using
  `inv_rms = rsqrtf(row_sum / HIDDEN_SIZE + EPS)`.
- EPS = **1e-5** pinned by the problem definition. This slug uses
  a larger epsilon than the rest of the cohort (the other plain
  RMSNorm slugs use 1e-6); the kernel mirrors the pinned value
  rather than blindly copying the sibling templates. Wider FP32
  accumulation is preserved per CLAUDE.md Rule 8.
- BF16 inputs / outputs, FP32 intermediates. No CUDA streams created
  by us, no `data_ptr`-keyed caching — kernel launches on
  `at::cuda::getCurrentCUDAStream()`.

`main.cpp` pins `hidden_size == 4096` via `TORCH_CHECK` so a workload
with a drifted hidden_size fails closed rather than silently degrading.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-checked
and natively-correctness-validated on. Final SOL scoring is on B200
(sm_100a). Differences worth noting for this pattern:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060): this kernel
  is bandwidth-bound (read `[B, 4096]` BF16, write `[B, 4096]` BF16,
  plus the small `[4096]` weight tile). B200's HBM3e (~8 TB/s) is
  roughly 8× the 3090 Ti's GDDR6X and ~5× the 5060's GDDR7, so local
  wall-time is a weak proxy for B200 SOL Score — the kernel will be
  even more bandwidth-dominant there.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are out of
  scope at Sm80 portability level; this kernel uses per-CTA shared
  memory and warp shuffles only.
- L2 carveout / partition: at modest batch sizes the entire row stays
  in L1/L2; for very large batches B200's larger L2 partition absorbs
  a working set the 3090 Ti's 6 MB L2 cannot.
- TCGen05 / TMEM divergence: 5th-gen tensor cores and TMEM register
  surface only exist on sm_100a; this kernel does not use tensor
  cores (the math is pointwise BF16/FP32 multiplies and adds), so the
  TCGen05 / TMEM split is not exercised here.
