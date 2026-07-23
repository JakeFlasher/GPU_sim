# Topic: Normalization (RMSNorm / LayerNorm / QK-Norm)

## What it covers

Fused RMSNorm, LayerNorm, and their variants (QK-norm, fused-add-norm,
norm + quant). Almost always memory-bound.

## Per-framework references

> **Note**: this table was pruned to the 5 kept frameworks during the solbench post-pivot trim. Rows for `sglang`, `vllm`, `tensorrt-llm`, `deepgemm`, `tilelang`, `cute-dsl`, `quack`, `tilekernels`, `veitner-blog`, `colfax-research`, `cuda-blog-kernels`, `thunderkittens`, `cccl-cub` were removed — query GitHub directly for those frameworks.

| Framework | Where to look | What it teaches |
| --- | --- | --- |
| `pytorch` | `aten/src/ATen/native/cuda/layer_norm_kernel.cu` | Correctness oracle. |
| `triton` | `python/tutorials/05-layer-norm.py` | Triton block-pointer norm reference. |

## Common optimization patterns

- Per-row CTA with block reduction; vectorized loads (`float4`, `bfloat162`).
- Fuse `add(residual, x)` before the reduction to avoid a second pass.
- Fuse the post-norm quantization (FP8 / FP4) into the same kernel.
- QK-norm: a small reduction per head; can share warp-level reductions
  across heads if hidden dim is small.

## Common bottlenecks

- HBM bandwidth (verify via counter-free surfaces: NVBit / CUPTI Activity / Nsys; `dram__throughput` NCU metric removed).
- Bank conflicts in shared-memory reductions when using non-warp-friendly
  vector widths.
- Long-scoreboard stalls when the reduction is forced into multiple passes.

