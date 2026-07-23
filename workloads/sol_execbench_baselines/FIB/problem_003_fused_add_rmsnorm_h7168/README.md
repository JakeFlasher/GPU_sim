# problem_003_fused_add_rmsnorm_h7168

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | Fused residual-add + RMSNorm at hidden_size=7168 for DeepSeek-V3; EPS=1e-6.                 |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_003_fused_add_rmsnorm_h7168.md`        |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/FlashInfer-Bench/003_fused_add_rmsnorm_h7168/`                |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Same proven fused-add-rmsnorm pattern as the smaller-hidden landings
(`problem_001_fused_add_rmsnorm_h2048`,
`problem_002_fused_add_rmsnorm_h4096`), scaled to the DeepSeek-V3
hidden size:

- One CTA per row (`[batch, hidden_size=7168]`).
- 512 threads per block, each handling ~7 BF16x2 pairs (the row has
  3584 BF16x2 pairs total).
- Two passes: sum-of-squares of `(hidden_states + residual)` in FP32
  with a warp-shuffle all-reduce then a single-warp cross-warp
  reduction; output rescale using
  `inv_rms = rsqrtf(row_sum / hidden_size + EPS)` and store
  `(h + r) * inv_rms * weight` in BF16.
- EPS pinned at the value the problem definition specifies (1e-6).
  Wider FP32 accumulation per CLAUDE.md Rule 8.
- BF16 inputs / outputs, FP32 intermediates. No CUDA streams created
  by us, no `data_ptr`-keyed caching, no monkey-patching — kernel
  reads only the current input pointers and writes only the
  destination-passing-style output pointer.

## Architecture-coupling caveats

`Ran-local-sm_86` evidence reflects local correctness on the RTX
3090 Ti host only. The B200 (`sm_100a`) path is unvalidated until
Phase 2. Differences worth noting for this pattern:

- HBM3e vs GDDR6X / GDDR7 bandwidth: this kernel is bandwidth-
  bound (read two `[B, 7168]` BF16 tensors + 1D weight, write
  `[B, 7168]` BF16). B200's HBM3e has roughly 20× the bandwidth
  of the 3090 Ti's GDDR6X and ~15× the bandwidth of the future
  cross-arch sm_120 RTX 5060's GDDR7, so local wall-time is a
  weak proxy for B200 SOL Score — the kernel will be even more
  bandwidth-dominant there.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are out
  of scope for sm_86; this kernel uses per-CTA shared memory and
  warp shuffles only.
- L2 carveout: at modest batch sizes the entire row stays in L1
  / L2; for very large batches B200's much larger L2 may absorb
  a working set the 3090 Ti's 6 MiB L2 cannot.
- TCGen05 / TMEM divergence: 5th-gen tensor cores and TMEM
  register surface only exist on sm_100a; this kernel does not
  use tensor cores (the math is pointwise BF16/FP32 multiplies
  and adds), so the TCGen05 / TMEM split is not exercised here.
- 7168 BF16 = 14 KiB per row of inputs; for the largest workload
  (B=256 rows) that is 3.5 MiB per tensor — fits in 3090 Ti L2
  with one or two rows of headroom.

## Calibration cohort

`predictor_friendly`. The access pattern is regular and the per-row
working set is small enough to fit comfortably in L1 across both
the sm_86 and sm_120 calibration hosts.
