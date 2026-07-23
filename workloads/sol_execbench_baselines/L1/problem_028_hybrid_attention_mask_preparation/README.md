# problem_028_hybrid_attention_mask_preparation

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | Hybrid attention-mask preparation: forward construction of a full causal mask + a sliding-window causal mask for an attention block that mixes full-attention layers and sliding-window-attention layers. Output dtype = bool. num_attention_heads=64, swa_num_attention_heads=64, sliding_window=128 pinned. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_028_hybrid_attention_mask_preparation.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/028_hybrid_attention_mask_preparation/`                    |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Compile-checked-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                             |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

One pure-boolean CUDA kernel that writes both the full causal
attention mask and the sliding-window attention mask in a single
pass over `(b, h, t, j)`.

Per-element math (matching the reference's two `masked_fill`
operations exactly):

```
Full causal mask:
  target_idx  = t
  source_idx  = j - past_kv_length
  causal_cond = (target_idx >= source_idx)
  full_mask = torch.ones(...).masked_fill(causal_cond, False)
    => full_attention_mask[b, h, t, j] = !causal_cond

Sliding-window mask:
  swa_mask = torch.zeros(...).masked_fill(valid_positions, False)
  This is a NO-OP: a buffer of zeros (False) filled with False at
  True positions yields zeros (False). The reference computes
  `valid_positions = causal_cond & window_cond` for clarity but
  the resulting tensor is always all-False because both the
  starting buffer AND the fill value are False.
    => sliding_window_attention_mask[b, h, t, j] = False
```

The kernel therefore writes `false` unconditionally for every
`sliding_window_attention_mask` element. The `sliding_window =
128` constant is accepted by the launcher for API-completeness
but is unused by the kernel — this matches the reference's
no-op semantics exactly.

### Kernel topology

- **Grid**: `(B, num_attention_heads=64, T)`. One CTA per `(b, h,
  t)` row.
- **BLOCK_THREADS = 256**. Stride-loop over `total_length`
  (= `seq_length + past_key_values_length`).
- Each thread `(b, h, t, j)` computes both bool outputs and
  writes them via contiguous row offsets.

### Runtime-check audit

const axes from `definition.json`:
- `num_attention_heads = 64`
- `swa_num_attention_heads = 64`
- `sliding_window = 128`

Runtime checks enforced before launch:

| Tensor                           | Runtime check                                              |
|----------------------------------|------------------------------------------------------------|
| `batch_size_scalar`              | `dim == 0`, `kInt64`; unboxed via `.item<int64_t>()` |
| `seq_length_scalar`              | `dim == 0`, `kInt64`; unboxed |
| `past_key_values_length_scalar`  | `dim == 0`, `kInt64`; unboxed |
| `full_attention_mask`            | `is_cuda`, `kBool`, `is_contiguous`, `dim == 4`, `size(0) == batch_size`, `size(1) == 64` (`num_attention_heads`), `size(2) == seq_length`, `size(3) == seq_length + past_kv_length` |
| `sliding_window_attention_mask`  | same checks, with `size(1) == 64` (`swa_num_attention_heads`) |

Destination-passing-style entry: both output mask tensors are
pre-allocated by the eval driver. The kernel writes each output
element exactly once.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-
checked and natively-correctness-validated on. Final SOL scoring
is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060):
  this kernel is bandwidth-bound on the bool output writes
  (`B * 64 * T * total_length` bytes per output, two outputs).
  B200's HBM3e roughly 8x bandwidth of the 3090 Ti's GDDR6X.
- DSMEM and cluster-cooperative writes (sm_100a-only) are out
  of scope at this Sm80-portability level.
- TCGen05 / TMEM divergence: this kernel uses no tensor cores
  and no fp computation.
- L2 carveout / partition: workload-dependent. The largest
  workload `[B=2, H=64, T=4096, total_len=8192]` has 2 * 64 *
  4096 * 8192 = 4 GiB per output — pure HBM streaming. The
  smallest workload `[B=2, H=64, T=32, total_len=128]` has
  2 * 64 * 32 * 128 = 512 KiB per output — fits in L2.
- Since the kernel has NO read traffic except the 3 host-side
  scalar inputs, all NCU read-only sector counters reflect only
  the implicit metadata reads CUDA emits for kernel launch
  bookkeeping. The kernel itself is pure-output.
