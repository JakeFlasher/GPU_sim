---
Use when: implementing grouped-query attention (GQA) with a paged KV cache during the decode phase — single-query attention against a long K/V history split into fixed-size pages. Covers head_dim ∈ {64, 128, 192, 256}, GQA ratios ∈ {2, 4, 8, 16}, page sizes ∈ {1, 16, 32}, BF16 / FP16 / FP8 inputs with FP32 reduction. Targets B200 sm_100a with sm_120 fallback; FlashInfer-family kernels in the FIB category instantiate this pattern.
Primary raw source: sol-execbench/examples/cutlass/
Supporting sources: references/NVIDIA_Manuals/cutlass_cute/; references/NVIDIA_Manuals/blackwell_tuning/; cutlass/include/cute/algorithm/copy.hpp
Repo touchpoints: cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl; cutlass/include/cute/atom/copy_atom.hpp; sol-execbench/examples/cutlass/; baselines/FIB/<problem_slug>/cutlass/; baselines/L1/<problem_slug>/cuda_cpp/
Patents anchored here: None
Out of scope:
  - Python / Triton / CuTe DSL Python / cuTile Python / cudnn_frontend Python / torch.compile submissions (CLAUDE.md Rule 5).
  - NCU / `nv-nsight-cu-cli` / `.ncu-rep`/`.ncu-repz` artefacts on any device, including local 5060 (CLAUDE.md Rule 1).
  - `nvidia-smi --lock-gpu-clocks` / `--lock-memory-clocks` / persistence / power / MIG / compute-mode mutation (CLAUDE.md Rule 2).
  - Privileged CUPTI: Events, Metrics, Range Profiling, PM Sampling, PerfWorks (CLAUDE.md Rule 3; CUPTI Activity remains allowed).
  - Host-driver work: NVIDIA driver install, DKMS, MOK, `modprobe nvidia`, kernel-module parameter toggles, reboot workflows (CLAUDE.md Rule 4).
  - User CUDA streams, background host threads, `torch.jit.fork`, null-stream side effects, CUDA Graphs without a per-problem note (CLAUDE.md Rule 7).
  - Evaluator-state exploits: `data_ptr` caching, lazy-eval `__eq__` on `FakeTensor`, monkey-patched `Event.elapsed_time` / `do_bench` (CLAUDE.md Rule 6).
  - Precision-contract violations: precision downgrade (e.g. FP16 compute upcast to FP32 output) to pass tolerance (CLAUDE.md Rule 8).
Version family: cuda-13.1; cutlass-4.5.x; sol-execbench-2026; flashinfer-2026
Default skill: implementing-sol-baselines
Routing tags: gqa; paged-attention; paged_attention; kv-cache; flashinfer; decode; attention; mqa; bf16; fp16; fp8
Latency authority: paper
---

# GQA Paged-KV Decode (FlashInfer-style)

## Use when

You're implementing the decode-phase attention for a SOL-ExecBench FIB-category problem. The query is a single new token (shape `[batch, num_q_heads, head_dim]`), the K/V cache is split into pages of fixed size (e.g. `page_size=1` means each token in its own page, `page_size=16` packs 16 tokens per page), and the query/key heads have a GQA grouping (multiple Q heads share one K head). The kernel must walk a `page_table` to gather the physical KV pages for each batch element, then run the QK^T → softmax → AV reduction.

## Canonical decomposition

```
batch_size × num_q_heads queries  ─┐
                                    │
page_table[batch, num_pages]  ─────►  gather K, V from paged storage
                                    │
                                    └─► fused: QK^T(p), softmax-online(p), A @ V(p)
                                          (one pass per page; running max & sum)
```

The decode kernel is online-softmax (Flash-style): keep `running_max`, `running_sum`, `running_output` per Q-head; for each page, compute the contribution and combine. The final output is `running_output / running_sum`.

## CUTLASS / CuTe building blocks

- **`cute::Layout`** for the paged KV view: `(batch, page_table_entries, page_size, num_kv_heads, head_dim)` with the `page_table` mapping logical position → physical page index.
- **`cutlass::TiledCopy`** with TMA descriptors (sm_100a) to stage one page at a time into shared memory. On sm_120 fall back to `cp.async` bulk loads.
- **`cutlass::TiledMma`** for QK^T (compute-bound at large GQA ratios) and AV (memory-bound from V re-read). On sm_100a use `tcgen05.mma` schedule; on sm_120 use `KernelMultistage`.
- **Online-softmax reduction** in shared memory; broadcast running max / sum across the GQA group (one K head feeds `gqa_ratio` Q heads).

Template skeleton (CUDA C++ surface, simplified):

```cpp
extern "C" __global__ void gqa_paged_decode_kernel(
    const __nv_bfloat16* __restrict__ Q,         // [B, H_q, D]
    const __nv_bfloat16* __restrict__ K_pages,   // [num_pages, page_size, H_kv, D]
    const __nv_bfloat16* __restrict__ V_pages,   // [num_pages, page_size, H_kv, D]
    const int32_t*       __restrict__ page_table,// [B, max_pages]
    const int32_t*       __restrict__ seq_lens,  // [B]
    __nv_bfloat16*       __restrict__ O,         // [B, H_q, D]
    float scale, int head_dim, int gqa_ratio,
    int page_size, int num_kv_heads)
{
    // 1) load Q[b, h_q, :] to registers
    // 2) for each page idx in page_table[b]:
    //      stage K_pages[idx], V_pages[idx] via TMA
    //      compute QK^T (single-row × page_size × head_dim)
    //      online-softmax update
    //      accumulate AV
    // 3) write O[b, h_q, :] = running_output / running_sum
}
```

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5060, GB206 die) compiles and smoke-runs; `sm_100a` (B200, GB100 die) is the scoring target. Concrete divergences:

- **TMA paged gather.** B200's TMA supports indirect descriptors that match the paged-KV access pattern naturally; sm_120 has a narrower TMA surface and the local kernel falls back to `cp.async` or scalar gathers. Bandwidth-bound decode performance differs substantially across arches.
- **TCGen05 / TMEM.** On sm_100a the QK^T and AV MMAs can accumulate in TMEM (the dedicated accumulator register file used by `tcgen05.mma`); on sm_120 accumulators live in regular registers, raising register pressure and lowering occupancy.
- **HBM3e vs GDDR7.** B200's HBM3e provides ~8 TB/s and decode is essentially memory-bound on the KV cache; the 5060's GDDR7 saturates much earlier and timing trends do NOT translate.
- **Cluster shapes.** B200 supports CTA clusters with DSMEM peer-read for the K-head broadcast across the GQA group; sm_120 does that broadcast through shared memory in a single CTA.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **page_table indirection on the wrong stream.** Default-stream discipline (CLAUDE.md Rule 7): launch on `at::cuda::getCurrentCUDAStream()`. The `page_table` is a small tensor and may tempt async copies on a side stream — don't.
- **`data_ptr` caching the K_pages base** (CLAUDE.md Rule 6). The harness shifts inputs by 256 B every iteration; the K_pages base address changes every call. Any memoised page-base address fails.
- **Wrong-precision RoPE.** When this pattern composes with the `rope_attention` pattern card, the K-rotation must run in FP32 angle space; downgrading to FP16 to share the FP16 MMA pipeline violates Rule 8.

## Linked problems

- `.claude/knowledge/problems/problem_012_gqa_paged_decode_h32_kv4_d128_ps1.md` — canonical FIB instantiation (32 Q heads, 4 KV heads, head_dim 128, page_size 1).
- Composes with `.claude/knowledge/patterns/rope_attention.md` when the model applies RoPE to K before paged-cache write.
