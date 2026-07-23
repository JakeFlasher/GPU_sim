---
Use when: implementing the MLP block with GELU / GEGLU activation (vs SwiGLU — see `swiglu_mlp.md` for that). Covers `y = down(gelu(x @ up))` (single-gate) and `y = down(gelu(x @ gate) * (x @ up))` (gated, GEGLU). BF16 / FP16 inputs, FP32 GELU. Targets B200 sm_100a with sm_120 fallback.
Primary raw source: cutlass/include/cutlass/epilogue/collective/sm100_epilogue_array_tma_warpspecialized.hpp
Supporting sources: references/NVIDIA_Manuals/cutlass_cute/; references/NVIDIA_Manuals/blackwell_tuning/
Repo touchpoints: cutlass/include/cutlass/epilogue/collective/sm100_epilogue_array_tma_warpspecialized.hpp; cutlass/include/cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp; baselines/L1/<problem_slug>/cutlass/; baselines/L2/<problem_slug>/cutlass/
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
Version family: cutlass-4.5.x; sol-execbench-2026
Default skill: implementing-sol-baselines
Routing tags: mlp; gelu; geglu; gate; gated-mlp; activation; tanh-approx-gelu; bf16; fp16
---

# Mlp Gelu Geglu

## Use when

Implementing the mlp block with gelu / geglu activation (vs swiglu — see `swiglu_mlp.md` for that). covers `y = down(gelu(x @ up))` (single-gate) and `y = down(gelu(x @ gate) * (x @ up))` (gated, geglu). bf16 / fp16 inputs, fp32 gelu. targets b200 sm_100a with sm_120 fallback.

## Canonical recipe

GELU MLP (non-gated):
```
y = down(gelu(x @ up))                       // two GEMMs + one elementwise
```
GEGLU (gated):
```
gate = x @ W_gate
up   = x @ W_up
y    = down(gelu(gate) * up)
```
GELU implementations: exact `0.5 * x * (1 + erf(x / sqrt(2)))` or the tanh approximation `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`. Match the spec's reference PyTorch.

## CUTLASS / CuTe / CUDA C++ building blocks

- `cutlass::gemm::device::GemmUniversal` for the two (or three) projections.
- `cutlass::epilogue::fusion::sm100_callbacks` for the fused-epilogue variant.
- FP32 GELU computation inside the epilogue callback; cast back to dtype.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5060, GB206 die) compiles and smoke-runs; `sm_100a` (B200, GB100 die) is the scoring target. Local pass on sm_120 verifies compile + correctness only; sm_100a is required for TCGen05 / TMA / DSMEM behavior, TMEM accumulation, and HBM3e bandwidth assumptions. Build twice when in doubt; the B200 problem packager auto-injects `-gencode arch=compute_100a,code=sm_100a`.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **Wrong GELU flavor** (exact vs tanh-approx) — verify against spec.
- **FP16 GELU upcast trick** is a Rule-8 violation.
- **Three projections on three streams** (Rule 7) — fused single-stream is required.

## Linked problems

See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `mlp_gelu_geglu` rows.
