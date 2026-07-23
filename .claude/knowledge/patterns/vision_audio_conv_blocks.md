---
Use when: implementing fused convolution + normalization + activation + residual blocks from vision / audio model families (ResNet, ViT patch embed, VAE encoder/decoder, audio Conv1D). Covers Conv2D / Conv3D / Conv1D with BF16 / FP16 / FP32 weights, GroupNorm / LayerNorm / RMSNorm fusion, SiLU / GELU / ReLU activations, optional residual add. Targets B200 sm_100a with sm_120 fallback.
Primary raw source: cutlass/include/cutlass/conv/collective/builders/sm100_umma_builder.inl
Supporting sources: references/NVIDIA_Manuals/cutlass_cute/; references/NVIDIA_Manuals/blackwell_tuning/; cutlass/include/cute/algorithm/copy.hpp
Repo touchpoints: cutlass/include/cutlass/conv/device/conv_universal_adapter.hpp; cutlass/include/cutlass/conv/collective/builders/sm100_umma_builder.inl; baselines/L1/<problem_slug>/cutlass/; baselines/L2/<problem_slug>/cutlass/
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
Routing tags: conv; conv2d; conv3d; conv1d; vision; audio; vae; resnet; implicit-gemm; groupnorm; layernorm; silu; gelu; bf16; fp16
---

# Vision Audio Conv Blocks

## Use when

Implementing fused convolution + normalization + activation + residual blocks from vision / audio model families (resnet, vit patch embed, vae encoder/decoder, audio conv1d). covers conv2d / conv3d / conv1d with bf16 / fp16 / fp32 weights, groupnorm / layernorm / rmsnorm fusion, silu / gelu / relu activations, optional residual add. targets b200 sm_100a with sm_120 fallback.

## Canonical recipe

Conv blocks decompose into:

1. Implicit-GEMM Conv via `cutlass::conv::device::ConvUniversal` (sm_100a) or the older `ImplicitGemmConvolution` interface for compatibility.
2. Fused norm / activation / residual via `cutlass::epilogue::fusion` callbacks chained after the implicit-GEMM accumulator.
3. For 3D conv (VAE), use `cutlass::conv::Conv3dProblemSize` and the `kAnalyticIDIGSwizzler` for the strided-GEMM transform.

## CUTLASS / CuTe / CUDA C++ building blocks

- `cutlass::conv::device::ConvUniversal` with `cutlass::conv::Operator::kFprop` (forward) / `kDgrad` / `kWgrad`.
- `cutlass::conv::collective::ConvCollective` builder using `sm100_implicit_gemm_umma_builder` for B200.
- `cutlass::epilogue::fusion::sm100_callbacks` chain to merge `groupnorm(silu(...))` into the conv output.
- For Conv3D, `cutlass::conv::ConvolutionMode::kCrossCorrelation` + 3D problem size.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5060, GB206 die) compiles and smoke-runs; `sm_100a` (B200, GB100 die) is the scoring target. Local pass on sm_120 verifies compile + correctness only; sm_100a is required for TCGen05 / TMA / DSMEM behavior, TMEM accumulation, and HBM3e bandwidth assumptions. Build twice when in doubt; the B200 problem packager auto-injects `-gencode arch=compute_100a,code=sm_100a`.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **Layout mismatch.** NCHW vs NHWC matters for the implicit-GEMM swizzler; verify against the spec's reference PyTorch.
- **GroupNorm in BF16** loses bits in the variance reduction; keep FP32 (Rule 8).
- **CUDA Graphs for the multi-launch sequence** is a Rule-7 reward-hack.

## Linked problems

See `tools/sol_baselines_scaffold/sol_pattern_mapping.md` `vision_audio_conv_blocks` rows.
