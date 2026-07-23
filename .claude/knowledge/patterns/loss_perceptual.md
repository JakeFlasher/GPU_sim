---
Use when: implementing a perceptual-loss feature-extraction kernel — typically a VGG-style backbone evaluated at multiple intermediate layers, summed with per-layer weights. Used in vision generation training (VAE, GAN). Targets B200 sm_100a with sm_120 fallback.
Primary raw source: cutlass/include/cutlass/conv/collective/builders/sm100_umma_builder.inl
Supporting sources: references/NVIDIA_Manuals/cutlass_cute/; references/NVIDIA_Manuals/blackwell_tuning/
Repo touchpoints: cutlass/include/cutlass/conv/device/conv_universal_adapter.hpp; baselines/L2/<problem_slug>/cutlass/
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
Routing tags: loss; perceptual-loss; vgg; feature-loss; lpips; conv-features; bf16; fp16; fp32
---

# Loss Perceptual

## Use when

Implementing a perceptual-loss feature-extraction kernel — typically a vgg-style backbone evaluated at multiple intermediate layers, summed with per-layer weights. used in vision generation training (vae, gan). targets b200 sm_100a with sm_120 fallback.

## Canonical recipe

```
for each layer L in vgg_backbone[1, 4, 7, ...]:
  f_real = vgg_L(real_img)
  f_fake = vgg_L(fake_img)
  loss_L = mean_l2(f_real - f_fake)
loss = sum_L(w_L * loss_L)
```
Conv blocks compose with `vision_audio_conv_blocks`; the per-layer L2 difference + sum is a reduction kernel.

## CUTLASS / CuTe / CUDA C++ building blocks

- `cutlass::conv` for the VGG forward passes.
- A `__global__` per-layer L2-mean kernel; FP32 accumulator.
- Final weighted-sum reduction across layers.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5060, GB206 die) compiles and smoke-runs; `sm_100a` (B200, GB100 die) is the scoring target. Local pass on sm_120 verifies compile + correctness only; sm_100a is required for TCGen05 / TMA / DSMEM behavior, TMEM accumulation, and HBM3e bandwidth assumptions. Build twice when in doubt; the B200 problem packager auto-injects `-gencode arch=compute_100a,code=sm_100a`.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **Caching VGG outputs** via `data_ptr` (Rule 6).
- **Weight-sum precision** — FP32 accumulator (Rule 8).
- **Multi-stream layer dispatch** (Rule 7).

## Linked problems

See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `loss_perceptual` rows.
