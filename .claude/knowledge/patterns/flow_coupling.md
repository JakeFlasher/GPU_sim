---
Use when: implementing a normalizing-flow coupling layer — partition input, apply an affine transformation conditioned on one half, recombine. Used in flow-based generative models (RealNVP, Glow). Covers BF16 / FP16 / FP32 inputs. Targets B200 sm_100a with sm_120 fallback.
Primary raw source: cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl
Supporting sources: references/NVIDIA_Manuals/cuda_guide/; references/NVIDIA_Manuals/cutlass_cute/
Repo touchpoints: baselines/L2/<problem_slug>/cuda_cpp/
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
Routing tags: flow; coupling; realnvp; glow; normalizing-flow; affine-coupling; bf16; fp16; fp32
---

# Flow Coupling

## Use when

Implementing a normalizing-flow coupling layer — partition input, apply an affine transformation conditioned on one half, recombine. used in flow-based generative models (realnvp, glow). covers bf16 / fp16 / fp32 inputs. targets b200 sm_100a with sm_120 fallback.

## Canonical recipe

```
x_a, x_b = split(x)                     // along channel dim
log_s, t = mlp(x_a)                     // affine parameters
y_a = x_a
y_b = x_b * exp(log_s) + t              // affine transform
y   = concat(y_a, y_b)
```
The MLP is a standard 2-3 layer block (see `mlp_gelu_geglu` or `swiglu_mlp`). The affine + concat step is memory-bound elementwise.

## CUTLASS / CuTe / CUDA C++ building blocks

- MLP sub-block (composes with `mlp_gelu_geglu`).
- A `__global__` elementwise kernel for the affine + concat step.
- FP32 internal compute for `exp(log_s)`.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5060, GB206 die) compiles and smoke-runs; `sm_100a` (B200, GB100 die) is the scoring target. Local pass on sm_120 verifies compile + correctness only; sm_100a is required for TCGen05 / TMA / DSMEM behavior, TMEM accumulation, and HBM3e bandwidth assumptions. Build twice when in doubt; the B200 problem packager auto-injects `-gencode arch=compute_100a,code=sm_100a`.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **`exp(log_s)` in BF16** is numerically unstable; FP32 is contractual.
- **Wrong split axis** — verify against spec.
- **Two-stream split** (Rule 7).

## Linked problems

See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `flow_coupling` rows.
