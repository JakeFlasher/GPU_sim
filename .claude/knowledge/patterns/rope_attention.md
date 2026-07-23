---
Use when: implementing rotary position embedding (RoPE) inside an attention kernel, including the KV-sharing variants used by GQA / MQA models. Covers the `q_rot = apply_rope(q, cos, sin)` and `k_rot = apply_rope(k, cos, sin)` rotations applied per-head before the QK^T product, on BF16 / FP16 inputs with FP32 angle math. Targets B200 sm_100a with sm_120 fallback.
Primary raw source: cutlass/include/cute/algorithm/copy.hpp
Supporting sources: references/NVIDIA_Manuals/cutlass_cute/; references/NVIDIA_Manuals/blackwell_tuning/; references/NVIDIA_Manuals/cuda_guide/
Repo touchpoints: cutlass/include/cute/atom/copy_atom.hpp; cutlass/include/cute/algorithm/copy.hpp; sol-execbench/examples/cuda_cpp/; baselines/L1/<problem_slug>/cuda_cpp/; baselines/L2/<problem_slug>/cuda_cpp/; baselines/FIB/<problem_slug>/cuda_cpp/
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
Version family: cuda-13.1; cutlass-4.5.x; sol-execbench-2026
Default skill: implementing-sol-baselines
Routing tags: rope; rotary; attention; kv-cache; kv-sharing; gqa; mqa; position-embedding; bf16; fp16
---

# RoPE Attention with KV Sharing

## Use when

You're implementing the rotary-position-embedding rotation inside or alongside an attention kernel for a SOL-ExecBench problem whose query/key tensors carry positional information through phase rotation rather than additive bias. RoPE is the standard in modern LLM families (LLaMA, Qwen, Mistral, DeepSeek) and pairs naturally with GQA / MQA (grouped / multi-query attention) where multiple Q heads share a single K head — the rotation must be applied per-head before the QK^T product, and the rotated K must be cached for KV-reuse decode.

## Canonical math

For each head-dim pair `(d_2i, d_2i+1)` and position `p`:

```
theta_i  = base^(-2i / head_dim)        // base typically 10000 or model-specific
cos_p_i  = cos(p * theta_i)
sin_p_i  = sin(p * theta_i)
q_rot[2i]   = q[2i] * cos_p_i - q[2i+1] * sin_p_i
q_rot[2i+1] = q[2i] * sin_p_i + q[2i+1] * cos_p_i
```

Same for `k`. The `cos` / `sin` tables are precomputed by the harness and passed as additional tensors. KV-sharing means `k_rot` is stored once and reused by every Q-head in its group during decode.

## Implementation surface

Two valid implementation shapes:

1. **Fused into the attention kernel.** Apply the rotation inside the producer warps that load Q and K into shared memory, before the TMA-staged QK^T contraction. This is the path most flagship FlashInfer-style kernels take on B200. The CUTLASS attention example (`sol-execbench/examples/cutlass/`) shows the structural template; the RoPE math is added in the prologue.
2. **Standalone elementwise kernel before attention.** Simpler to author: two CUDA C++ kernels (one for Q, one for K) writing rotated tensors that the attention kernel consumes as inputs. Pays an extra HBM round-trip but is correct and easy to verify on sm_120.

For a Round 1 baseline, start with shape (2) and reserve (1) for a perf-tuning pass.

## CUTLASS / CuTe building blocks

- `cute::Layout` to walk the `(seq_len, num_heads, head_dim)` tensor in head-dim pairs.
- `cute::TiledCopy` with TMA descriptors on sm_100a for `cos` / `sin` tables (small, fits in shared memory or constant cache).
- `cutlass::FastDivmod` for the per-position theta index when the position offset is non-trivial (e.g. paged-KV decode where positions are scattered).
- FP32 sin / cos for accuracy; cast Q / K back to the input dtype after rotation. **Do NOT downgrade Q / K precision to FP16 to fit a faster MMA path** — that is the Rule 8 precision-downgrade reward-hack.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5060, GB206 die) is the compile and smoke target; `sm_100a` (cloud B200, GB100 die) is the scoring target. Concrete divergences:

- **TMA-staged Q/K load.** B200 sm_100a TMA descriptors can prefetch the cos / sin tables into shared memory cheaply; sm_120 lacks the full TMA surface and the local kernel falls back to `cp.async` or scalar loads. Local timing on the 5060 is NOT a proxy for B200's TMA-staged path.
- **WGMMA / TCGen05.** When fusing RoPE into a TCGen05-scheduled attention kernel, the rotation prologue must complete before the producer warps issue `tcgen05.mma`; sm_120 uses `KernelMultistage` and the prologue runs in the multistage producer warp instead.
- **head_dim broadcast.** B200's larger L2 lets the cos / sin tables stay resident across an entire batch; sm_120's smaller L2 may evict them. Bandwidth-bound shape regimes can look different across arches.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **KV-cache write order.** Apply RoPE to K BEFORE writing to the paged KV cache; the cache must store the rotated K so decode reads do not double-rotate.
- **Default-stream discipline** (CLAUDE.md Rule 7). Launch on `at::cuda::getCurrentCUDAStream()`; do not enqueue the cos / sin tabulation on a separate stream.
- **`data_ptr` caching** (CLAUDE.md Rule 6). The harness shifts input pointers by 256 B every iteration — any data_ptr-keyed memoisation of the rotated K (or of the cos / sin tables themselves) goes stale and fails correctness.

## Linked problems

- FIB-category RoPE-bearing kernels: `problem_012_gqa_paged_decode_h32_kv4_d128_ps1.md` exercises the RoPE-on-decoded-K path inside the GQA paged decode (the pattern card and the dossier are complementary: this card holds the rotation recipe; the dossier holds problem-shape constraints).
- L1 / L2 RoPE-bearing attention problems link here via the matching-pattern field of their dossiers under `.claude/knowledge/problems/`.
