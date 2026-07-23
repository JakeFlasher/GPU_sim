---
Use when: implementing standalone elementwise activations — SiLU, GELU, ReLU, Sigmoid, Tanh, Hardswish, etc. — not fused into a GEMM epilogue. Covers BF16 / FP16 / FP32 inputs with optional bias add. Targets B200 sm_100a with sm_120 fallback.
Primary raw source: references/NVIDIA_Manuals/cuda_guide/00_index.md
Supporting sources: references/NVIDIA_Manuals/ptx_isa/; cutlass/include/cute/algorithm/copy.hpp
Repo touchpoints: baselines/L1/<problem_slug>/cuda_cpp/
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
Routing tags: activation; elementwise; silu; gelu; relu; sigmoid; tanh; hardswish; bf16; fp16; fp32
---

# Activation Elementwise

## Use when

Implementing standalone elementwise activations — silu, gelu, relu, sigmoid, tanh, hardswish, etc. — not fused into a gemm epilogue. covers bf16 / fp16 / fp32 inputs with optional bias add. targets b200 sm_100a with sm_120 fallback.

## Canonical recipe

Memory-bound elementwise — pure HBM-bandwidth kernels:
```
y[i] = activation(x[i])              // optionally with bias
```
Use 128-bit vectorized loads / stores via `int4` casts; FP32 internal computation; cast back to input dtype for write.

## CUTLASS / CuTe / CUDA C++ building blocks

- A `__global__` elementwise kernel; vectorized `int4` loads.
- FP32 activation computation; `__expf`, `__tanhf` intrinsics under fast-math.
- `cute::TiledCopy` for vectorized writes on sm_100a.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5060, GB206 die) compiles and smoke-runs; `sm_100a` (B200, GB100 die) is the scoring target. Local pass on sm_120 verifies compile + correctness only; sm_100a is required for TCGen05 / TMA / DSMEM behavior, TMEM accumulation, and HBM3e bandwidth assumptions. Build twice when in doubt; the B200 problem packager auto-injects `-gencode arch=compute_100a,code=sm_100a`.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **FP16 activation upcast trick** is a Rule-8 reward-hack.
- **Vector-load mis-alignment** — assert `data_ptr % 16 == 0` only when verified.
- **Stream split across input tiles** is a Rule-7 violation.

## Linked problems

See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `activation_elementwise` rows.
