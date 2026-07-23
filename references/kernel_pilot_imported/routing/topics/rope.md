# Topic: RoPE / Rotary Position Embeddings

## What it covers

Rotary position embedding kernels: standard RoPE, NeoX-style interleaved,
GPT-J-style paired, YaRN / NTK extensions, long-context variants.

## Per-framework references

> **Note**: this table was pruned to the 5 kept frameworks during the solbench post-pivot trim. Rows for `sglang`, `vllm`, `tensorrt-llm`, `deepgemm`, `tilelang`, `cute-dsl`, `quack`, `tilekernels`, `veitner-blog`, `colfax-research`, `cuda-blog-kernels`, `thunderkittens`, `cccl-cub` were removed — query GitHub directly for those frameworks.

| Framework | Where to look | What it teaches |
| --- | --- | --- |
| `flashinfer` | `include/flashinfer/pos_enc.cuh` | Fused RoPE inside the attention kernel. |
| `triton` | community implementations referenced from SGLang / vLLM Triton paths | Triton block-pointer RoPE templates. |

## Common optimization patterns

- Apply RoPE in-place on Q / K right before the attention kernel; avoids a
  second pass.
- Vectorized cos/sin LUT loads.
- Fuse RoPE with QK-norm when both are applied per head.
- For long-context (NTK / YaRN), precompute the frequency tensor once and
  index it by position.

## Common bottlenecks

- Memory-bound; verify via counter-free surfaces (NVBit / CUPTI Activity / Nsys; `dram__throughput` NCU metric removed).
- Mismatched interleaving (NeoX vs GPT-J) is a frequent source of silent
  correctness drift; always test against `pytorch` reference.

