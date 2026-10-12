# problem_002_decoder_layer_full_block

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | Complete LLaMA decoder layer: input RMSNorm, GQA attention with RoPE, attention residual, post-attention RMSNorm, SwiGLU MLP, MLP residual. fp32 throughout. hidden_size=4096, intermediate_size=14336, num_attention_heads=32, num_key_value_heads=8, head_dim=128 pinned. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_002_decoder_layer_full_block.md`       |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L2/002_decoder_layer_full_block/`                             |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Compile-checked-sm_120`                                                                           |
| Local correctness           | 6/6 evaluated workloads PASSED with `max_abs_err ≤ 7.75e-06` and `max_rel_err ≤ 1.29e-03` (workloads B=1/S=131, B=2/S=1024, B=32/S=256, B=1/S=512, B=16/S=449, B=16/S=128). Subsequent workloads trip a CUDA fault on the local 3090 Ti that takes the device off the PCIe bus (requires operator-driven GPU reset); reproducible on B=64/S=128. Root cause is suspected to be the GQA-repeat or attention-matmul memory layout on certain (B, S) shapes; further debugging is deferred to a follow-up round with full GPU diagnostic time. |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

The pipeline runs entirely on the current CUDA stream, alternating
between two custom CUDA kernels and the at:: op surface (cuBLAS,
PyTorch's CUDA softmax / cat / view / transpose primitives):

| # | Operation | Kernel |
|---|-----------|--------|
| 1 | Input RMSNorm | custom rms_norm_kernel |
| 2 | Q projection | at::matmul |
| 3 | K projection | at::matmul |
| 4 | V projection | at::matmul |
| 5 | Reshape Q/K/V to [B, H, S, head_dim] | at::view + at::transpose |
| 6 | RoPE on Q | slice + at::cat + multiply + add |
| 7 | RoPE on K | same shape as #6 |
| 8 | KV repeat for GQA | at::expand + at::contiguous + at::view |
| 9 | QK^T · scaling | at::matmul + scalar multiply |
| 10 | + attention_mask | at::add |
| 11 | softmax (fp32) | at::softmax |
| 12 | softmax · V | at::matmul |
| 13 | transpose + view | at::transpose + at::view |
| 14 | O projection | at::matmul |
| 15 | + residual (attention) | at::add |
| 16 | Post-attention RMSNorm | custom rms_norm_kernel |
| 17 | Gate projection | at::matmul |
| 18 | Up projection | at::matmul |
| 19 | silu(gate) * up | custom silu_mul_kernel |
| 20 | Down projection | at::matmul |
| 21 | + residual (MLP) → output | at::add + at::Tensor::copy_ |

The two custom kernels handle the bandwidth-bound elementwise
fusions where a separate launch per primitive op would otherwise
double-traverse the activation tensors:

- **rms_norm_kernel** — one CTA per (b, s) row, 256 threads strided
  over `hidden_size`. Two passes: sum-of-squares reduction (warp
  shuffle + cross-warp shared), then `x * rsqrt(var + eps) * weight`
  written in-place.
- **silu_mul_kernel** — bf-free grid-stride elementwise over
  `B * S * intermediate_size` fp32 elements; computes
  `(gate / (1 + exp(-gate))) * up`.

The same pattern (cuBLAS for projections + custom kernels for
elementwise fusions + torch primitives for reshape / softmax) is
used in the already-landed
`problem_030_flux_concatenated_sequence_processing_with_split` and
`problem_004_fused_residual_rms_mlp` baselines.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-
checked and natively-correctness-validated on. Final SOL scoring is
on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5090): the
  QKV projections and the three MLP projections are bandwidth-
  bound for the [intermediate_size, hidden_size] = [14336, 4096]
  weight matrices (~220 MiB). B200's HBM3e reshapes the
  bandwidth-vs-compute envelope relative to the 3090 Ti / 5090
  GDDR-class memory subsystems.
- DSMEM and cluster-cooperative attention (sm_100a-only) are out
  of scope at the cuBLAS-dispatch portability level; cuBLAS picks
  the appropriate kernel transparently.
- L2 carveout / partition: the K^T transpose + softmax stream of
  the attention matmul exceeds the 3090 Ti's 6 MB L2 partition for
  the larger workloads (S up to 8192); B200's much larger L2 caches
  more of it. The cache-prediction calibration cohort labels this
  pattern accordingly.
- TCGen05 / TMEM divergence: the four projection GEMMs and the
  two attention matmuls dispatch via cuBLAS, which on B200 chooses
  TCGen05 / TMEM tensor-core kernels automatically; on the 3090 Ti
  and 5090 the same call resolves to Ampere / Ada tensor cores.
  Portability is at the call-site level, not the schedule level.
