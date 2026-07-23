# Topic: Activation / Element-wise Fusion

## What it covers

Fused activation kernels (silu, gelu, swiglu), fused-add-residual,
fused-dequant-activation. Almost always memory-bound; the value comes from
fusing them into a neighbor compute kernel.

## Per-framework references

> **Note**: this table was pruned to the 5 kept frameworks during the solbench post-pivot trim. Rows for `sglang`, `vllm`, `tensorrt-llm`, `deepgemm`, `tilelang`, `cute-dsl`, `quack`, `tilekernels`, `veitner-blog`, `colfax-research`, `cuda-blog-kernels`, `thunderkittens`, `cccl-cub` were removed — query GitHub directly for those frameworks.

| Framework | Where to look | What it teaches |
| --- | --- | --- |
| `pytorch` | `aten/src/ATen/native/cuda/Activation.cu` | Correctness oracle. |
| `triton` | `python/tutorials/02-fused-softmax.py` style templates | Triton fused activation reference. |

## Common optimization patterns

- Always fuse into the preceding or following GEMM epilogue when possible.
- Vectorize loads / stores (`float4`, `bfloat162`, `half2`).
- For swiglu, fuse the two-projection `silu(x) * y` into one pass.

## Common bottlenecks

- Standalone activation kernels are pure-bandwidth and should be fused.
- Misaligned tail handling can degrade vectorization on the last tile.

