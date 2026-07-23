---
Use when: choosing `cute::TiledMma` and the underlying `MMA_Atom` for a GEMM-shaped kernel. Determines which tensor-core instruction (`tcgen05.mma`, `wgmma.mma`, classic `mma`) the inner loop emits, and how warps are partitioned across the MMA shape. Referenced by every GEMM / attention pattern card.
Primary raw source: cutlass/include/cute/atom/mma_atom.hpp
Supporting sources: cutlass/include/cute/atom/mma_traits_sm100.hpp; references/NVIDIA_Manuals/cutlass_cute/; references/NVIDIA_Manuals/ptx_isa/
Repo touchpoints: cutlass/include/cute/atom/mma_atom.hpp; cutlass/include/cute/atom/mma_traits.hpp
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
Routing tags: tiled-mma; tiledmma; mma-atom; cute; tensor-core; tcgen05; gemm; attention
---

# TiledMma (CUTLASS Construct)

## Use when

Choosing the MMA atom that the CollectiveMainloop will issue. The `cute::TiledMma` wraps an `MMA_Atom` and a partition layout that maps threads → MMA fragments.

## Canonical recipe

```cpp
using TiledMma = cute::TiledMma<
    cute::MMA_Atom<cutlass::arch::Sm100_QMMA_F32_BF16_BF16_F32_SS>,
    cute::Layout<cute::Shape<_2, _2, _1>>,    // partition across warps
    cute::Tile<_128, _128, _64>                // outer MMA tile
>;
```

Atom families:
- **Sm100_QMMA_F32_BF16_BF16_F32_SS** — sm_100a BF16 × BF16 → FP32 via `tcgen05.mma`.
- **Sm100_QMMA_F32_E4M3_E4M3_F32_SS** — sm_100a FP8 e4m3.
- **Sm100_QMMA_F32_E2M1_E2M1_F32_SS** — sm_100a NVFP4 e2m1 block-scaled.
- **Sm90_64x64x16_F32BF16BF16F32_SS** — Hopper-style WGMMA atom (sm_120 fallback).

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

The MMA atom is arch-conditional: `Sm100_QMMA_*` atoms emit `tcgen05.mma` and require sm_100a; on sm_120 the corresponding `Sm89_QMMA_*` or `Sm90_*` atoms emit `mma.sync` / `wgmma` instead. The CollectiveBuilder picks the right atom from the template parameters; do NOT hardcode a single atom across both targets.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this construct card.

## Pattern cards that compose this construct

`gemm_baseline`, `attention_core`, `attention_backward`, `flash_attention`, `fp8_blockwise_quant`, `mlp_gelu_geglu`, `swiglu_mlp`, `gqa_paged_decode`, `gqa_paged_prefill`, `mla_paged_attention`, `transformer_layer`, `moe_dispatch`, `vision_audio_conv_blocks`, `loss_perceptual`.
