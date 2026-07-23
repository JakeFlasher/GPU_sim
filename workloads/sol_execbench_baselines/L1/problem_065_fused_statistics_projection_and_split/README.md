# problem_065_fused_statistics_projection_and_split

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | fp32 fused Conv1D projection to 2× channels followed by channel-wise split into mean and log-variance statistics. Captured from OpenVoice TTS encoder. `in_channels = 192`, `out_channels = 192`, `kernel_size = 1`, `out_channels_2 = 384` pinned. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_065_fused_statistics_projection_and_split.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/065_fused_statistics_projection_and_split/`                |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Pure ATen/cuBLAS dispatch — NO custom CUDA kernels. The reference
algorithm decomposes naturally into one Conv1D + a channel-wise
split:

    stats = F.conv1d(x, weight, bias, padding=0)
            # [B, 384, T]
    mean, logs = torch.split(stats, 192, dim=1)

For `kernel_size=1` fp32 inputs on Sm80, PyTorch's `at::conv1d`
dispatches to a cuDNN implicit-gemm kernel that reduces to a
`cublasSgemm` at the equivalent GEMM shape (M = B*T, K = 192,
N = 384). The result matches the reference's `F.conv1d`
bit-for-bit because both code paths route through the same cuDNN
+ cuBLAS dispatch.

Implementation in `main.cpp`:

1. **`at::conv1d(x, weight, bias, stride=1, padding=0,
   dilation=1, groups=1)`** → `stats[B, 384, T]` workspace
   (allocated implicitly by `at::conv1d`).

2. **`mean.copy_(stats.narrow(1, 0, 192))`** — fills the
   pre-allocated DPS `mean[B, 192, T]` with the first half of
   `stats` along the channel dimension.

3. **`logs.copy_(stats.narrow(1, 192, 192))`** — fills the
   pre-allocated DPS `logs[B, 192, T]` with the second half.

The two `copy_` calls are bandwidth-bound element-wise tensor
copies (ATen dispatch). No custom CUDA kernel surface; this is
the cohort's simplest landing pattern so far.

### Allocations

`stats[B, 384, T]` is allocated implicitly by `at::conv1d`.
The two outputs `mean[B, 192, T]` and `logs[B, 192, T]` are
pre-allocated by the eval driver (DPS) and filled via the
`copy_` calls.

### Runtime-check audit

Every const axis from `definition.json` enforced on every input
and output tensor that the kernel indexes with that fixed
assumption:

| Tensor   | Runtime check                                              |
|----------|------------------------------------------------------------|
| `x`      | `is_cuda`, `kFloat32`, `is_contiguous`, `dim == 3`, `size(1) == in_channels = 192` |
| `weight` | same dtype/contig, `dim == 3`, `size(0) == out_channels_2 = 384`, `size(1) == in_channels = 192`, `size(2) == kernel_size = 1` |
| `bias`   | same dtype/contig, `dim == 1`, `size(0) == out_channels_2 = 384` |
| `mean`   | same dtype/contig, `dim == 3`, `size(0) == x.size(0)`, `size(1) == out_channels = 192`, `size(2) == x.size(2)` |
| `logs`   | same dtype/contig, `dim == 3`, `size(0) == x.size(0)`, `size(1) == out_channels = 192`, `size(2) == x.size(2)` |

Destination-passing-style entry: both `mean` and `logs` are
pre-allocated by the eval driver. Each output is written
exactly once by `narrow().copy_()`.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was
compile-checked and natively-correctness-validated on. Final
SOL scoring is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060):
  the `weight` tensor (`384 × 192 × 4 = 288 KiB` fp32) fits
  comfortably in the L1 cache on every target arch; the
  `stats` workspace (`B × 384 × T × 4` bytes — up to ~50 MiB
  at the largest workload) exceeds the 3090 Ti's 6 MiB L2,
  so the `narrow().copy_()` reads from HBM/GDDR rather than
  L2 for large workloads.
- DSMEM and cluster-cooperative tile staging (sm_100a-only)
  are out of scope at this Sm80-portability level.
- TCGen05 / TMEM divergence: the 1×1 fp32 conv runs on the
  FP32 SM pipe via cuBLAS (tf32 mma on Ampere; B200's
  TCGen05 / TMEM is NOT exercised at this dtype).
- L2 carveout / partition: workload-dependent. cuDNN may
  pick a different algorithm on B200 because of its larger
  L2; the cuBLAS sgemm path the cohort-B probe pins is the
  Sm80 sgemm dispatch.
- The two `narrow().copy_()` calls are bandwidth-bound on
  the half-channel-stripe copy from `stats` into `mean` /
  `logs`. Each copy reads/writes `B × 192 × T × 4 = 0.75 ×
  stats_bytes` of bandwidth. The two copies could in
  principle be fused with the conv into a single launch
  that writes the two halves directly, but that would
  require a custom CUDA kernel — out of scope for this
  pure-ATen-dispatch landing.
