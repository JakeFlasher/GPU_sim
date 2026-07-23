# Topic: Sampling / Speculative Decode

## What it covers

Logits processing (temperature, top-k, top-p, min-p, penalty), speculative /
tree decode bookkeeping (EAGLE / Medusa / draft-target verification), greedy
sampling.

## Per-framework references

> **Note**: this table was pruned to the 5 kept frameworks during the solbench post-pivot trim. Rows for `sglang`, `vllm`, `tensorrt-llm`, `deepgemm`, `tilelang`, `cute-dsl`, `quack`, `tilekernels`, `veitner-blog`, `colfax-research`, `cuda-blog-kernels`, `thunderkittens`, `cccl-cub` were removed — query GitHub directly for those frameworks.

| Framework | Where to look | What it teaches |
| --- | --- | --- |
| `flashinfer` | `include/flashinfer/sampling.cuh`, `python/flashinfer/sampling.py` | Top-k / top-p / min-p / temperature sampling primitives. |

## Common optimization patterns

- Two-pass top-k: warp-level partial top-k, then block-level merge.
- Fused top-k + softmax for top-p sampling.
- Tree-decode: gather logits via a per-branch index tensor, then verify with
  a small kernel that compares draft tokens to target tokens.
- For very-small vocab cases, prefer a one-pass radix-style top-k.

## Common bottlenecks

- `atomicAdd` contention on the candidate token buffer (avoid with warp
  reductions).
- Long-scoreboard stalls when reading the logits tensor without vectorized
  loads.
- Tree-decode gather is gather-heavy; consider sharded layouts.

