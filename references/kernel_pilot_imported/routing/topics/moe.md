# Topic: Mixture of Experts (MoE)

## What it covers

MoE kernels: top-k routing, `moe_align_block_size`, permute / unpermute,
grouped GEMM, fused activation, expert-parallel comm.

## Per-framework references

> **Note**: this table was pruned to the 5 kept frameworks during the solbench post-pivot trim. Rows for `sglang`, `vllm`, `tensorrt-llm`, `deepgemm`, `tilelang`, `cute-dsl`, `quack`, `tilekernels`, `veitner-blog`, `colfax-research`, `cuda-blog-kernels`, `thunderkittens`, `cccl-cub` were removed — query GitHub directly for those frameworks.

| Framework | Where to look | What it teaches |
| --- | --- | --- |
| `triton` | `python/tutorials/` + community kernels | Triton block-pointer fused-MoE templates. |

## Common optimization patterns

- **`moe_align_block_size`**: sort tokens by expert and pad each expert's
  block to a fixed BLOCK_SIZE so the grouped GEMM has aligned tiles. Compute
  the permutation map without a global sort by using a cumulative-sum trick.
- **Grouped GEMM**: concatenate per-expert GEMMs along M with a
  `group_offsets` array; the kernel iterates per group.
- **Fused activation in epilogue**: silu / swiglu / gelu fused into the
  grouped GEMM epilogue.
- **Expert-parallel routing**: combine `top-k` routing with all-to-all
  bookkeeping; overlap comm with intermediate GEMM.

## Common bottlenecks

- Permute / unpermute are gather-heavy and often hit shared-memory bank
  conflicts.
- Tail batches: when one expert receives many more tokens than the others,
  SMs become idle.
- Top-k softmax on the routing logits can be a surprising bottleneck for
  large expert counts.

