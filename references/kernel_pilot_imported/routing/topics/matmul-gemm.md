# Topic: GEMM / Tensor-Core matmul

## What it covers

Dense matmul kernels including dense FP16 / BF16 GEMM, FP8 / FP4 GEMM,
block-scaled GEMM, grouped GEMM, GEMV (skinny matmul), Stream-K / Split-K,
fused epilogues (bias, residual, activation, dequant).

## Per-framework references

> **Note**: this table was pruned to the 5 kept frameworks during the solbench post-pivot trim. Rows for `sglang`, `vllm`, `tensorrt-llm`, `deepgemm`, `tilelang`, `cute-dsl`, `quack`, `tilekernels`, `veitner-blog`, `colfax-research`, `cuda-blog-kernels`, `thunderkittens`, `cccl-cub` were removed — query GitHub directly for those frameworks.

| Framework | Where to look | What it teaches |
| --- | --- | --- |
| `cutlass` | `include/cutlass/gemm/`, `examples/48_hopper_*`, `examples/60_*`, `tools/profiler/` | The canonical Hopper / Blackwell GEMM templates with TMA + WGMMA. |
| `triton` | `python/tutorials/03-matrix-multiplication.py`, `python/tutorials/09-persistent-matmul.py` | Block-pointer + autotune matmul reference. |
| `pytorch` | `aten/src/ATen/native/cuda/Blas.cpp`, `torch/_inductor/kernel/mm.py` | cuBLAS / Inductor matmul as a baseline. |

## Common optimization patterns

- Hopper TMA loads with warpgroup-specialized producer / consumer.
- Persistent kernel grid (one CTA per SM, loops over output tiles).
- Stream-K for low-batch or skinny-K shapes.
- Fused epilogue: bias + activation + residual + dequant in one pass.
- Block-scaled FP8: 128x128 weight scale, 1x128 activation scale (Hopper).
- Cluster-launched cooperative GEMM on Hopper / Blackwell.

## Common bottlenecks

- Skinny-K decode is memory-bound; tensor pipe underutilized.
- Tail batches in grouped GEMM cause SM imbalance; use Stream-K or padding.
- Epilogue dominates total time when activation / dequant is heavy; fuse it.

