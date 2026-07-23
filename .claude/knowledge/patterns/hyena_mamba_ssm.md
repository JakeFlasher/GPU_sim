---
Use when: implementing selective-scan / state-space model (SSM) kernels — Mamba selective scan, Hyena long-convolution + FFT, Mamba2 SSD. Covers BF16 / FP16 / FP32 inputs with FP32 state accumulation and chunked / parallel scan algorithms. Targets B200 sm_100a with sm_120 fallback.
Primary raw source: cutlass/include/cute/algorithm/copy.hpp
Supporting sources: references/NVIDIA_Manuals/cuda_guide/; references/NVIDIA_Manuals/blackwell_tuning/; references/NVIDIA_Manuals/cutlass_cute/
Repo touchpoints: cutlass/include/cute/atom/copy_atom.hpp; cutlass/include/cute/algorithm/copy.hpp; baselines/L1/<problem_slug>/cuda_cpp/; baselines/L2/<problem_slug>/cuda_cpp/
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
Routing tags: ssm; mamba; mamba2; hyena; selective-scan; ssd; state-space; scan; recurrence; bf16; fp16
---

# Hyena Mamba Ssm

## Use when

Implementing selective-scan / state-space model (ssm) kernels — mamba selective scan, hyena long-convolution + fft, mamba2 ssd. covers bf16 / fp16 / fp32 inputs with fp32 state accumulation and chunked / parallel scan algorithms. targets b200 sm_100a with sm_120 fallback.

## Canonical recipe

Selective scan:

```
h_t = exp(Δ_t · A) · h_{t-1} + (Δ_t · B_t) · x_t        // state update
y_t = C_t · h_t + D · x_t                                 // output projection
```
with `A` diagonal, `B_t, C_t` input-dependent, `Δ_t` softplus-gated. Implementations:

1. **Chunk-parallel scan.** Within a chunk run the recurrence sequentially in registers; between chunks combine via prefix-sum-of-`(exp(ΔA), Δ·B·x)` pairs.
2. **Hyena.** The convolution branch becomes an FFT (`cuFFT`) when the kernel is long; the gating branch is a standard MLP.
3. **Mamba2 SSD** reinterprets the scan as a structured matrix-multiply, amenable to `cutlass::gemm`.

## CUTLASS / CuTe / CUDA C++ building blocks

- A `__global__` chunk-parallel scan kernel in CUDA C++; FP32 state accumulators.
- `cutlass::gemm::device::GemmUniversal` for Mamba2 SSD's structured-matrix step.
- `cuFFT` plan caching (host-side) for Hyena's long-convolution branch.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5060, GB206 die) compiles and smoke-runs; `sm_100a` (B200, GB100 die) is the scoring target. Local pass on sm_120 verifies compile + correctness only; sm_100a is required for TCGen05 / TMA / DSMEM behavior, TMEM accumulation, and HBM3e bandwidth assumptions. Build twice when in doubt; the B200 problem packager auto-injects `-gencode arch=compute_100a,code=sm_100a`.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **State precision** — accumulating `h_t` in BF16 diverges over long sequences; FP32 is mandatory.
- **`data_ptr` on the state buffer** breaks under the 256-byte shift (Rule 6).
- **FFT plan cached by tensor address** silently goes stale; key by shape only.

## Linked problems

See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `hyena_mamba_ssm` rows.
