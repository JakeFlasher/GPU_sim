# Topic: KV-Cache / Paged Memory

## What it covers

Paged KV layouts (block tables, page tables), HiCache / RadixAttention
prefix sharing, KV swap / copy / reshape kernels, append-attention bookkeeping.

## Per-framework references

> **Note**: this table was pruned to the 5 kept frameworks during the solbench post-pivot trim. Rows for `sglang`, `vllm`, `tensorrt-llm`, `deepgemm`, `tilelang`, `cute-dsl`, `quack`, `tilekernels`, `veitner-blog`, `colfax-research`, `cuda-blog-kernels`, `thunderkittens`, `cccl-cub` were removed — query GitHub directly for those frameworks.

| Framework | Where to look | What it teaches |
| --- | --- | --- |
| `flashinfer` | `include/flashinfer/page.cuh`, `python/flashinfer/*` (BatchPrefill/BatchDecode wrappers) | Flexible page table with variable block sizes. |

## Common optimization patterns

- Page-aligned KV blocks with a per-request block-index tensor.
- Prefix-cache aware attention: split shared prefix from unique suffix to
  avoid redundant KV loads.
- KV reshape / swap: avoid Python-side reshapes; the kernel writes directly
  into the paged layout.
- HiCache: host-resident eviction tier with async H2D for warm prefixes.

## Common bottlenecks

- Indirect KV loads through block tables stress L2 and HBM; check
  [NCU metrics removed; use counter-free surfaces from solbench's scripts/baselines_harness/ for L2/L1 traffic estimation].
- KV append on extend-attention can become a host-overhead bottleneck if
  Python wrappers re-allocate.

