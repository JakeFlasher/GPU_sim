---
Use when: implementing DeepSeek-style Multi-head Latent Attention (MLA) — paged attention with a compressed KV cache (latent KV split into low-rank `c_kv` and a RoPE'd k_pe component). Covers BF16 / FP16 inputs, head_dim families 16-heads × `ckv=512 + kpe=64`. Targets B200 sm_100a with sm_120 fallback.
Primary raw source: sol-execbench/examples/cutlass/
Supporting sources: references/NVIDIA_Manuals/cutlass_cute/; references/NVIDIA_Manuals/blackwell_tuning/; cutlass/include/cute/algorithm/copy.hpp
Repo touchpoints: cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl; baselines/FIB/<problem_slug>/cutlass/
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
Routing tags: mla; multi-head-latent-attention; deepseek; latent-kv; paged-attention; paged_attention; bf16; fp16
---

# Mla Paged Attention

## Use when

Implementing deepseek-style multi-head latent attention (mla) — paged attention with a compressed kv cache (latent kv split into low-rank `c_kv` and a rope'd k_pe component). covers bf16 / fp16 inputs, head_dim families 16-heads × `ckv=512 + kpe=64`. targets b200 sm_100a with sm_120 fallback.

## Canonical recipe

MLA splits each KV head into:

- `c_kv ∈ [seq, 512]` — low-rank latent (no positional info; non-rotated).
- `k_pe ∈ [seq, 64]`  — small RoPE'd positional component.
Decode flow:
```
Q_c = q_proj_c(Q)                            // project Q into c_kv space
Q_pe = rope(q_proj_pe(Q))                    // project + rotate Q's pe part
S    = Q_c @ c_kv.T + Q_pe @ k_pe.T          // two GEMMs summed
P    = softmax(S * scale)
O    = P @ c_kv                              // single AV with the latent value
```
Save HBM vs vanilla GQA because `c_kv` is shared across all heads and only the small `k_pe` carries position info.

## CUTLASS / CuTe / CUDA C++ building blocks

- Two `cutlass::gemm` collectives for the dual `Q_c @ c_kv.T` and `Q_pe @ k_pe.T` reduction.
- `rope_attention` building blocks for the Q_pe rotation prologue.
- Paged-KV gather (same primitives as `gqa_paged_decode`).

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5060, GB206 die) compiles and smoke-runs; `sm_100a` (B200, GB100 die) is the scoring target. Local pass on sm_120 verifies compile + correctness only; sm_100a is required for TCGen05 / TMA / DSMEM behavior, TMEM accumulation, and HBM3e bandwidth assumptions. Build twice when in doubt; the B200 problem packager auto-injects `-gencode arch=compute_100a,code=sm_100a`.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **Wrong split** — `c_kv` and `k_pe` must be cached separately under the page layout.
- **RoPE on `c_kv`** is incorrect; only `k_pe` gets rotated.
- **Two streams for the dual reduction** is a Rule-7 violation.

## Linked problems

See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `mla_paged_attention` rows.
