---
Use when: computing or refreshing the RoPE cos / sin frequency tables — `theta_i = base^(-2i/head_dim)`, `cos_p_i / sin_p_i` for positions `p`. Distinct from the rotation kernel (`rope_attention.md`); this card covers table-only kernels and the dynamic-NTK / YaRN scaling variants. Targets B200 sm_100a with sm_120 fallback.
Primary raw source: references/NVIDIA_Manuals/cuda_guide/00_index.md
Supporting sources: references/NVIDIA_Manuals/cutlass_cute/; cutlass/include/cute/algorithm/copy.hpp
Repo touchpoints: baselines/L1/<problem_slug>/cuda_cpp/; baselines/L2/<problem_slug>/cuda_cpp/
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
Version family: cuda-13.1; sol-execbench-2026
Default skill: implementing-sol-baselines
Routing tags: rope; rope-tables; rope-cos-sin; theta-table; dynamic-ntk; yarn; position-embedding; fp32
---

# Rope Embedding Tables

## Use when

Computing or refreshing the rope cos / sin frequency tables — `theta_i = base^(-2i/head_dim)`, `cos_p_i / sin_p_i` for positions `p`. distinct from the rotation kernel (`rope_attention.md`); this card covers table-only kernels and the dynamic-ntk / yarn scaling variants. targets b200 sm_100a with sm_120 fallback.

## Canonical recipe

```
theta_i  = base^(-2i / head_dim)            // i in [0, head_dim/2)
cos_p_i  = cos(p * theta_i)
sin_p_i  = sin(p * theta_i)
```
Compute in FP32; cast to BF16 / FP16 if the downstream rotation kernel consumes the cast dtype. Dynamic-NTK and YaRN modify `theta_i` based on context-length scaling factors before the table is built.

## CUTLASS / CuTe / CUDA C++ building blocks

- A `__global__` table-build kernel in CUDA C++; FP32 trig in fast-math mode.
- `__sinf` / `__cosf` intrinsics with `--use_fast_math` for speed.
- `cute::TiledCopy` for the final dtype-cast write to HBM on sm_100a.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5060, GB206 die) compiles and smoke-runs; `sm_100a` (B200, GB100 die) is the scoring target. Local pass on sm_120 verifies compile + correctness only; sm_100a is required for TCGen05 / TMA / DSMEM behavior, TMEM accumulation, and HBM3e bandwidth assumptions. Build twice when in doubt; the B200 problem packager auto-injects `-gencode arch=compute_100a,code=sm_100a`.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **Cached table base address** via `data_ptr` (Rule 6) — the harness shifts inputs.
- **Wrong base value.** Common values: `10000.0` (LLaMA), `1e6` (Qwen with YaRN), per-model.
- **Precision downgrade.** Computing `theta` in BF16 loses 1-2 bits per multiply; FP32 is contractual.

## Linked problems

See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `rope_embedding_tables` rows.
