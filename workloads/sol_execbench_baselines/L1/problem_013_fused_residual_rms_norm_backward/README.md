# problem_013_fused_residual_rms_norm_backward

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | Backward pass for fused residual + RMSNorm; outputs grad_hidden_states (bf16), grad_residual (bf16), grad_weight (fp32). hidden_size=2560 pinned. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_013_fused_residual_rms_norm_backward.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/013_fused_residual_rms_norm_backward/`                     |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Compile-checked-sm_120`                                                                           |
| Local correctness           | 15/16 workloads PASSED; workload #1 (B=64, S=8192) is INVALID_REFERENCE — the PyTorch reference creates 5 GiB intermediates that exceed the 3090 Ti's 24 GiB VRAM. Not a kernel-correctness fault. |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Two-kernel decomposition of the analytic backward formula:

```
row_sum[b, s]   = sum_h grad_output[h] * weight[h] * normalized[h]
mean_grad_norm  = row_sum / H
grad_x[b, s, h] = rstd[b, s] * (grad_output[h] * weight[h]
                                  - mean_grad_norm * normalized[h])
grad_weight[h]  = sum_{b, s} grad_output[b, s, h] * normalized[b, s, h]
```

- **Kernel A — per-row grad_x.** One CTA per (b, s) row; 256 threads
  grid-stride along the contiguous H axis (H=2560). Two passes over
  the row: first to accumulate row_sum via warp-shuffle + cross-warp
  shared-memory reduction, second to write grad_x (cast to bf16) into
  both grad_hidden_states and grad_residual. The reference makes
  grad_residual a `.clone()` of grad_x_bf16, so the two outputs are
  bit-identical and a single FP32→BF16 conversion feeds both stores.
- **Kernel B — per-column grad_weight.** One CTA per h ∈ [0, H);
  256 threads grid-stride across the B·S rows at fixed h. Block-wide
  warp + cross-warp reduction, then thread 0 stores `grad_weight[h]`.
  Strided global reads on the local 3090 Ti — this is one of the
  predictor-hostile access patterns the cache-prediction calibration
  needs to label.
- The saved-forward tensor `x` is part of the function signature but
  is unused by the reference math; the kernel takes it as `const&` to
  preserve argument-order parity and ignores its contents.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-
checked and natively-correctness-validated on. Final SOL scoring is
on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060): Kernel B
  is the per-column reduction with strided loads; B200's HBM3e
  bandwidth plus the larger L2 partition will cache far more of the
  B·S strided stream than the 3090 Ti's GDDR6X / 6 MB L2 can. On the
  sm_120 5060 the GDDR7 bandwidth lands between the two but the L2
  capacity is comparable to the 3090 Ti.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are out of
  scope; both kernels reduce strictly per-CTA via warp shuffles and
  per-CTA shared memory.
- L2 carveout / cache: the per-column kernel is the predictor-hostile
  cohort case — its strided access pattern across [B·S, H] interacts
  with L2 carveout differently on each host.
- TCGen05 / TMEM divergence: this kernel does not use tensor cores;
  the math is fp32 elementwise + reduction. The TCGen05 / TMEM split
  is not exercised at all.
