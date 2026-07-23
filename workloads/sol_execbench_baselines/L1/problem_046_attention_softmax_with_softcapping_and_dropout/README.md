# problem_046_attention_softmax_with_softcapping_and_dropout

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | BF16 attention softmax with Gemma3 logit softcapping (`tanh(x/30) * 30` clamp to ≈ ±30), fp32-upcast numerical-stability softmax over the last (K) axis. Captured from Gemma-3-1B-it. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_046_attention_softmax_with_softcapping_and_dropout.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/046_attention_softmax_with_softcapping_and_dropout/`       |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

CUDA C++ device-side per-row attention softmax with the Gemma3
softcap transformation fused inline.

- One CTA per row over `[batch_size, num_heads=8, seq_len_q,
  seq_len_k]` (CTA index → `batch_size * num_heads * seq_len_q`
  rows). Each CTA processes one row of `seq_len_k` BF16 values.
- BLOCK_THREADS = 256 with stride-loop over the row;
  warp-shuffle + cross-warp shared-memory reduction for max
  then sum.
- Three streaming passes over the input row:
  1. Find row max of `softcap(x) = tanh(x/30) * 30`.
  2. Compute `sum(exp(softcap(x) - row_max))`.
  3. Write `exp(softcap(x) - row_max) / row_sum` as bf16.
- All reductions run in fp32 for numerical stability per the
  reference's `dtype=torch.float32` upcast. BF16 inputs/outputs.
- The softcap formula is recomputed each pass rather than cached
  in shared memory — for typical `seq_len_k` (the workload sweep
  covers up to ~1024) the recompute cost is dominated by the
  global BF16 load.
- `main.cpp` pins `num_heads == 8` via `TORCH_CHECK`.
- No CUDA streams created by us, no `data_ptr`-keyed caching —
  kernel launches on `at::cuda::getCurrentCUDAStream()`.

The "dropout" in the problem name reflects the broader Gemma-2/3
attention block context; the actual kernel under measurement is
just the softcap + softmax (the reference's `@torch.no_grad()`
inference path does not exercise dropout).

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-checked
and natively-correctness-validated on. Final SOL scoring is on B200
(sm_100a). Differences worth noting for this pattern:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060): this
  kernel is bandwidth-bound (one read of `seq_len_k` BF16 elements
  per row, repeated 3× across the passes, with one write of the
  same row). B200's HBM3e (~8 TB/s) lifts the per-row throughput
  ceiling ~8× over the 3090 Ti's GDDR6X.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are out
  of scope at Sm80 portability level. The per-CTA reduction here
  uses warp shuffle + per-CTA shared memory only.
- L2 carveout / partition: at modest `(B, H, Q, K)` the input row
  fits in L1 / L2; for large `seq_len_k` (e.g. 4096+) the row's
  bf16 working set crosses the 3090 Ti's 6 MB L2 boundary and
  every pass goes to HBM. B200's larger L2 absorbs much larger
  rows.
- TCGen05 / TMEM divergence: 5th-gen tensor cores and TMEM
  register surface only exist on sm_100a; this kernel does not
  use tensor cores (the math is pointwise tanh/exp/multiplies
  plus reductions), so the TCGen05 / TMEM split is not
  exercised here.
