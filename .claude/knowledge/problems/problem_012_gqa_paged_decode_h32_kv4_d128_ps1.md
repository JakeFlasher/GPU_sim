---
Use when: starting the FIB-category baseline for `problem_012_gqa_paged_decode_h32_kv4_d128_ps1` -- grouped-query attention with paged KV cache, decode phase. 32 query heads, 4 KV heads (GQA ratio 8), head_dim 128, page_size 1. FlashInfer-family inference primitive.
Primary raw source: references/NVIDIA_Manuals/solbench_problems/problem_012_gqa_paged_decode_h32_kv4_d128_ps1.md
Supporting sources: .claude/knowledge/patterns/gemm_baseline.md; .claude/knowledge/sources/blackwell-tuning-docs.md; .claude/knowledge/sources/cutlass-cute-docs.md
Repo touchpoints: baselines/FIB/problem_012_gqa_paged_decode_h32_kv4_d128_ps1/cutlass/; sol-execbench/examples/cutlass/gemm/; cutlass/include/cutlass/gemm/collective/
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
Version family: sol-execbench-2026; flashinfer-2026
Default skill: implementing-sol-baselines
Routing tags: gqa; paged-attention; paged_attention; problem_012; flashinfer; decode; kv-cache
---

# problem_012_gqa_paged_decode_h32_kv4_d128_ps1

## Use when

Implementing a decode-phase grouped-query attention kernel with a paged KV cache. This is a FlashInfer-family (FIB) inference primitive; the SOL-ExecBench FIB category has 26 such primitives traced from production LLM serving systems (vLLM, SGLang).

## Workload summary

| Field | Value |
|---|---|
| Operation | Decode-phase GQA with paged KV cache; per-token softmax over historical KV blocks |
| Heads | 32 query heads, 4 KV heads (GQA ratio = 8) |
| Head dim | 128 |
| Page size | 1 (per the slug suffix `ps1`) |
| Dtypes | BF16 or FP16 Q/K/V; FP32 attention-score accumulator per CLAUDE.md Rule 8 |
| Category | FIB (FlashInfer primitive) |

## Matching patterns

GQA paged-decode is its own pattern family — the dedicated pattern card is `.claude/knowledge/patterns/gqa_paged_decode.md`, which captures the paged-KV gather, the online softmax, and the GQA broadcast across query heads. The per-head QK^T and attention-output matmuls also compose `.claude/knowledge/patterns/gemm_baseline.md`.

## Implementation entry point

Author under `baselines/FIB/problem_012_gqa_paged_decode_h32_kv4_d128_ps1/cutlass/` (or `cuda_cpp/`). Entry point per `sol-execbench/docs/solution.md`:

```cpp
void run(torch::Tensor query, torch::Tensor kv_cache,
         torch::Tensor page_table, torch::Tensor sequence_lengths,
         torch::Tensor output) {
    // GQA paged-decode attention; output written in-place
}
```

(See the canonical spec for exact tensor names and the page-table layout.)

## Algorithmic structure

1. **Page gather.** Walk the per-sequence page table to materialise the KV span for this query (single decode token per sequence).
2. **QK^T per query head.** Each KV head fans out to 8 query heads (GQA ratio); broadcast the K projection across the query-head group.
3. **Softmax.** FP32 accumulator across the historical KV span; numerically stable subtract-max + exp.
4. **Attention-output matmul.** `output = attn_weights @ V` where V is gathered from the paged cache.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (RTX 5060) and `sm_100a` (B200) divergences for paged GQA decode:

- **TMA for KV-page gather (sm_100a).** B200 can issue a TMA descriptor for the gather of an arbitrary page sequence into shared memory in a single bulk async-copy. sm_120 must use the older `cp.async` per-element paths.
- **DSMEM cluster-cooperative softmax.** B200 can split the KV-history reduction across a CTA cluster with DSMEM peer-loads when the history is long. sm_120 must keep the reduction inside a single CTA, raising shared-memory pressure on long sequences.
- **Memory bandwidth.** Paged-attention is memory-bound for short KV histories (decode is the worst case); B200's HBM3e gives a much higher upper bound than sm_120's GDDR7.

Evidence-label vocabulary per `CLAUDE.md` lives in `baselines/FIB/problem_012_gqa_paged_decode_h32_kv4_d128_ps1/README.md`, NOT in this dossier.

## Cross-references

- Source spec: `references/NVIDIA_Manuals/solbench_problems/problem_012_gqa_paged_decode_h32_kv4_d128_ps1.md`.
- Pattern recipe: `.claude/knowledge/patterns/gqa_paged_decode.md` (primary) and `.claude/knowledge/patterns/gemm_baseline.md` (for the per-head GEMMs).
- Blackwell-specific tuning: `.claude/knowledge/sources/blackwell-tuning-docs.md`.
