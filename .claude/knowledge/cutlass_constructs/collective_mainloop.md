---
Use when: choosing the CollectiveMainloop builder for a CUTLASS GEMM / Conv / Attention kernel — the producer-consumer warp-specialised inner-loop schedule that drives TMA / `cp.async` staging and `tcgen05.mma` issue on sm_100a (with `KernelMultistage` fallback on sm_120). Used by every GEMM-shaped pattern card in the atlas.
Primary raw source: cutlass/include/cutlass/gemm/collective/collective_builder.hpp
Supporting sources: cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl; cutlass/include/cute/algorithm/copy.hpp; references/NVIDIA_Manuals/cutlass_cute/
Repo touchpoints: cutlass/include/cutlass/gemm/collective/collective_builder.hpp; cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl
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
Routing tags: collective; collective-mainloop; mainloop; collective-builder; cutlass; schedule; tcgen05; tma; gemm
---

# CollectiveMainloop (CUTLASS Construct)

## Use when

Selecting the CollectiveMainloop builder for a GEMM-shaped kernel. The builder picks a kernel schedule (cooperative warp-specialised, pingpong, multistage), a pipeline depth, and the right TMA / `cp.async` staging policy for the target arch.

## Canonical recipe

```cpp
using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            sizeof(typename CollectiveEpilogue::SharedStorage)>,
        cutlass::gemm::collective::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;
```

On sm_100a the builder picks the `KernelTmaWarpSpecializedCooperative` schedule (TMA-staged producer warps + `tcgen05.mma` consumer warps with TMEM accumulator). On sm_120 the same template falls back to `KernelMultistage` (Hopper-style two-stage pipeline).

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

The CollectiveMainloop's *schedule* is sm_100a-conditional: TCGen05 / TMA-staged producer / TMEM accumulation are sm_100a-only. On sm_120 the builder degrades automatically to `KernelMultistage` which uses regular registers and `cp.async`. A local compile-check pass on sm_120 verifies template instantiation; it does NOT validate the sm_100a schedule's pipeline behaviour.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this construct card.

## Pattern cards that compose this construct

`gemm_baseline`, `attention_core`, `attention_backward`, `flash_attention`, `mlp_gelu_geglu`, `swiglu_mlp`, `gqa_paged_decode`, `gqa_paged_prefill`, `mla_paged_attention`, `transformer_layer`, `moe_dispatch` — every GEMM-shaped pattern goes through a CollectiveMainloop.
