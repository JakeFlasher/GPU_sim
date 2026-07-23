# Topic: Quantization (FP8 / FP4 / INT8 / AWQ / GPTQ)

## What it covers

Low-precision GEMM, dequant-on-the-fly kernels, block-scaled FP8 / FP4
layouts, AWQ / GPTQ INT4 / INT8 weight-only kernels, SmoothQuant W8A8.

## Per-framework references

> **Note**: this table was pruned to the 5 kept frameworks during the solbench post-pivot trim. Rows for `sglang`, `vllm`, `tensorrt-llm`, `deepgemm`, `tilelang`, `cute-dsl`, `quack`, `tilekernels`, `veitner-blog`, `colfax-research`, `cuda-blog-kernels`, `thunderkittens`, `cccl-cub` were removed — query GitHub directly for those frameworks.

| Framework | Where to look | What it teaches |
| --- | --- | --- |
| `cutlass` | `examples/54_hopper_fp8_*`, `examples/60_blackwell_*` | Reference Hopper / Blackwell FP8 / FP4 GEMM templates. |

## Common optimization patterns

- **Block-scaled FP8**: 128x128 weight scale, 1x128 activation scale, scale
  tensors stored next to the data.
- **Per-channel + per-token scaling**: scales fused into the GEMM epilogue.
- **Weight-only INT4 (AWQ / GPTQ)**: dequant-on-the-fly into registers,
  matmul as FP16 / BF16. Skinny-batch decode is the common target.
- **Pre-quant fusion**: combine activation, quant, and scale into one kernel
  before the GEMM.

## Common bottlenecks

- Scale-tensor stride mismatch causes silent correctness drift.
- AWQ / GPTQ kernels are sensitive to the `group_size` / `M` ratio; benchmark
  across the full decode batch range.
- FP8 paths fall back to a slow path silently when alignment is missing.

