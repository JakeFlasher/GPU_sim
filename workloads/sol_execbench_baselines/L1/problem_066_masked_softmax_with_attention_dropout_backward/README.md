# problem_066_masked_softmax_with_attention_dropout_backward

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | fp32 backward through dropout(masked_softmax(x)): dropout backward + softmax backward + masked_fill backward over a `[B, H, T, T]` attention-score tensor with `[B, 1, T, T]` mask broadcast across the head axis. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_066_masked_softmax_with_attention_dropout_backward.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/066_masked_softmax_with_attention_dropout_backward/`       |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Compile-checked-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                             |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Single CUDA C++ kernel that fuses dropout-backward + softmax-
backward + masked_fill-backward in one per-row CTA pass over the
softmax dimension.

Algorithm per row `(b, h, t)`:

```
(a) dropout backward:
    if p_dropout > 0:
      grad_softmax_output[j] = grad_output[j] * dropout_mask[j] / (1 - p_dropout)
    else:
      grad_softmax_output[j] = grad_output[j]

(b) softmax backward (derived from Jacobian J_ij = y_i * (delta_ij - y_j)):
    sum_term = sum_j (p_attn[j] * grad_softmax_output[j])
    grad_softmax_input[j] = p_attn[j] * (grad_softmax_output[j] - sum_term)

(c) masked_fill backward (zero gradient where mask == False):
    grad_scores[j] = mask[b, 0, t, j] ? grad_softmax_input[j] : 0
```

### Kernel topology

- **Grid**: `(B, H, T)`. One CTA per `(b, h, t)` row.
- **BLOCK_THREADS = 256** (8 warps of 32 lanes). Stride-loop over
  the row length `T` (which can exceed BLOCK_THREADS).
- **Pass 1** (reduction): each thread accumulates a fp32 partial
  sum over its stride of `(b, h, t, j)`: `p_attn[j] *
  grad_softmax_output[j]`. Warp-shuffle reduce within each warp;
  cross-warp tree reduce via 8-element shared-memory array;
  thread 0 broadcasts the final `sum_term` back through shared
  memory.
- **Pass 2** (write): each thread re-reads `grad_output[j]`,
  `dropout_mask[j]`, `p_attn[j]`, `mask[b, 0, t, j]` and writes
  `grad_scores[j] = mask ? p_attn[j] * (grad_softmax_output[j] -
  sum_term) : 0`.

The two passes are necessary because `sum_term` must be known
before computing the per-element `grad_scores`. Reading the
row twice from global is acceptable for fp32 contiguous tensors;
the second pass is L1-warm from the first pass for small T (T <
BLOCK_THREADS * 4 row elements fit in per-SM L1 cache).

### Mask broadcast

`mask` is shape `[B, 1, T, T]` and broadcasts across the head
axis. The kernel reads `mask[b, 0, t, j]` at offset
`b * T * T + t * T + j`, sharing the same `b` and `t` across all
heads — so each (b, t) row of mask is read `H` times per launch.

### Runtime-check audit

No const axes are pinned by `definition.json`. Runtime checks
reduce to cross-tensor shape coherence:

| Tensor         | Runtime check                                              |
|----------------|------------------------------------------------------------|
| `grad_output`  | `is_cuda`, `kFloat32`, `is_contiguous`, `dim == 4`, last two dims equal |
| `p_attn`       | `is_cuda`, `kFloat32`, `is_contiguous`, `.sizes() == grad_output.sizes()` |
| `mask`         | `is_cuda`, `kBool`, `is_contiguous`, `dim == 4`, sizes `(B, 1, T, T)` matching grad_output's B and T |
| `dropout_mask` | `is_cuda`, `kBool`, `is_contiguous`, `.sizes() == grad_output.sizes()` |
| `grad_scores`  | `is_cuda`, `kFloat32`, `is_contiguous`, `.sizes() == grad_output.sizes()` |

Destination-passing-style entry: `grad_scores` is pre-allocated
by the eval driver. The kernel writes each output element exactly
once.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-
checked and natively-correctness-validated on. Final SOL scoring
is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5090): B200's
  HBM3e roughly 8x bandwidth of the 3090 Ti's GDDR6X. This kernel
  is bandwidth-bound on the two passes; B200 reads each row's
  fp32 traffic ~8x faster.
- DSMEM and cluster-cooperative cross-row staging (sm_100a-only)
  are out of scope at this Sm80-portability level.
- TCGen05 / TMEM divergence: this kernel uses no tensor cores;
  the B200 TCGen05 / TMEM path is NOT exercised here.
- L2 carveout / partition: workload-dependent. The largest
  workload `[B=1, H=32, T=512]` has 32 * 512 * 512 * 4 B = 32 MiB
  per fp32 tensor; the four fp32 inputs total ~128 MiB which
  exceeds the 3090 Ti's 6 MiB L2 — pure HBM streaming for the
  bulk. For smaller workloads `[B=2, H=4, T=256]` totalling
  ~2 MiB per fp32 tensor, L2 captures cross-pass reuse.
