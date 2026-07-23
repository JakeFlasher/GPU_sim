---
Use when: choosing or composing CUTLASS `EpilogueVisitor` callbacks — the post-MMA fusion chain that applies bias / activation / residual-add / norm / scale / dtype-cast / aux-store inside the MMA epilogue, avoiding an HBM round-trip. Referenced by every MLP / norm / fused-attention pattern in the atlas.
Primary raw source: cutlass/include/cutlass/epilogue/collective/sm100_epilogue_array_tma_warpspecialized.hpp
Supporting sources: cutlass/include/cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp; cutlass/include/cutlass/epilogue/fusion/sm100_visitor_compute_tma_warpspecialized.hpp; references/NVIDIA_Manuals/cutlass_cute/
Repo touchpoints: cutlass/include/cutlass/epilogue/collective/sm100_epilogue_array_tma_warpspecialized.hpp; cutlass/include/cutlass/epilogue/fusion/
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
Routing tags: epilogue; epilogue-visitor; epilogue-builder; fusion; callbacks; aux-store; bias; activation
---

# EpilogueVisitor (CUTLASS Construct)

## Use when

Composing the fused epilogue chain for a GEMM-shaped kernel. The `EpilogueVisitor` lets you express `output = activation(bias + alpha * acc + beta * residual)` as a callback tree rather than three separate kernels.

## Canonical recipe

```cpp
using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::TmaWarpSpecializedCooperative,
        FusionCallbacks  // user-defined tree of fusion::Sm100EVT* nodes
    >::CollectiveOp;
```

Common fusion nodes:
- `Sm100EVT<Compute, Op>` — apply a unary op (SiLU, GELU, sigmoid).
- `Sm100Compute<Op, ElementOut, ElementIn>` — leaf compute node.
- `Sm100AuxLoad` — load an auxiliary tensor (e.g. `gate` for SwiGLU; bias).
- `Sm100AuxStore` — store an auxiliary tensor (e.g. saving normalised activations for later loss).

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

The fused-epilogue path is sm_100a's main win: callbacks run on warp-specialised consumer warps using TMEM accumulator, so the activation fusion costs near-zero extra HBM. On sm_120 the same callbacks compile but run in regular registers, with reduced fusion benefit. Plan baseline numbers around this gap.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this construct card.

## Pattern cards that compose this construct

`gemm_baseline`, `attention_core`, `attention_backward`, `mlp_gelu_geglu`, `swiglu_mlp`, `transformer_layer`, `fp8_blockwise_quant`, `embedding_norm_projection`, `vision_audio_conv_blocks`, `sam_mask_decoder`.
