---
Use when: implementing a CUTLASS C++ GEMM baseline kernel for a SOL-ExecBench L1 / L2 problem (single-op or fused matmul). Covers shape classes N=128 to N=28672, dtypes BF16 / FP16 / FP32 / FP8, and the canonical TiledMma + CollectiveMainloop + EpilogueVisitor stack on B200 (sm_100a) with sm_120 fallback.
Primary raw source: cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl
Supporting sources: cutlass/examples/cute/tutorial/; sol-execbench/examples/cutlass/gemm/; references/NVIDIA_Manuals/cutlass_cute/; references/NVIDIA_Manuals/blackwell_tuning/
Repo touchpoints: cutlass/include/cutlass/gemm/device/gemm_universal.h; cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl; cutlass/include/cute/atom/mma_atom.hpp; baselines/L1/<problem_slug>/cutlass/
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
Routing tags: gemm; matmul; cutlass; tiled-mma; collective; tensor-core; tcgen05; bf16; fp8
Latency authority: paper
---

# GEMM Baseline (Canonical CUTLASS Recipe)

## Use when

You're starting a SOL-ExecBench problem whose primary operation is a dense matrix multiply (e.g. `problem_004_gemm_n128_k2048`, `problem_007_gemm_n4096_k4096`, `problem_011_gemm_n28672_k4096`). Single-op L1 GEMMs and the matmul-bearing portions of L2 fused kernels both start here.

## Canonical CUTLASS recipe

Use `cutlass::gemm::device::GemmUniversal` with a sm_100a-targeted collective builder:

1. **CollectiveMainloop** -- pick from `cutlass::gemm::collective::sm100_collective_builder<...>` (CUTLASS 3.8+). For BF16 inputs with FP32 accumulator the typical tile shapes are 128x128x64 through 256x128x64; for FP8 e4m3 / e5m2 inputs the block-scaled variants apply (see `fp8_blockwise_quant` pattern card).
2. **TiledMma** -- `cute::TiledMma<cute::MMA_Atom<cutlass::arch::Sm100_QMMA_F32_BF16_BF16_F32_SS>, ...>` or the FP16 / FP8 analogues.
3. **EpilogueVisitor** -- `cutlass::epilogue::collective::sm100_epilogue_builder<...>` with optional fusion (residual add, RMSNorm, activation). For pure GEMMs the linear epilogue is sufficient.
4. **Schedule** -- `cutlass::gemm::KernelTmaWarpSpecializedCooperative` for warp-specialized producer / consumer pipeline; falls back to `KernelMultistage` on sm_120 where TCGen05 instructions are unavailable.

Template instantiation skeleton:

```cpp
using Gemm = cutlass::gemm::device::GemmUniversal<
    cutlass::bfloat16_t,                         // ElementA
    cutlass::layout::RowMajor,                   // LayoutA
    cutlass::bfloat16_t,                         // ElementB
    cutlass::layout::ColumnMajor,                // LayoutB
    cutlass::bfloat16_t,                         // ElementC
    cutlass::layout::RowMajor,                   // LayoutC
    float,                                       // ElementAccumulator (FP32 required)
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm100,                        // sm_100a for B200
    cutlass::gemm::collective::CollectiveBuilder</* tile, cluster, pipeline */>,
    cutlass::epilogue::collective::EpilogueBuilder</* ... */>
>;
```

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

`sm_120` (local RTX 5060, GB206 die) is the compile and smoke target; `sm_100a` (cloud B200, GB100 die) is the scoring target. Divergences for GEMM:

- **Tensor-core opcode availability.** TCGen05 instructions (`tcgen05.mma`, `tcgen05.cp`) are sm_100a-only. On sm_120 the warp-specialised TCGen05 schedule falls back to the multistage Hopper-style pipeline (`KernelMultistage`).
- **TMEM allocations.** TMEM is the sm_100a accumulator-residence register file used by `tcgen05.mma`. On sm_120 accumulators stay in the regular register file; CUTLASS's collective builder handles the divergence when the schedule policy is chosen correctly.
- **Cluster shapes.** B200 supports up to 16-CTA clusters with DSMEM peer-read; sm_120 has smaller cluster limits. Do not hardcode `ClusterShape<_2, _2, _1>` unless verified on both arches -- let the collective builder choose.
- **HBM3e vs GDDR7 bandwidth.** B200 has dramatically higher memory bandwidth than the 5060 (paper authority: 8 TB/s on B200 per `references/NVIDIA_Manuals/blackwell_tuning/`); memory-bound shape regimes look very different and sm_120 wall-time is NOT a proxy for sm_100a SOL Score.
- **L2 partitioning.** B200's L2 is significantly larger than the 5060's; tile-size selection that fits B200's L2 may thrash the 5060's L2.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this pattern card.

## Common pitfalls

- **Precision downgrade is a reward-hack** (CLAUDE.md Rule 8). Do not perform FP16 compute and upcast to FP32 output to pass tolerance -- the harness detects this.
- **Default-stream discipline** (CLAUDE.md Rule 7). Launch on `at::cuda::getCurrentCUDAStream()`; do not create user streams or use `cudaGraph` without a per-problem note.
- **No `data_ptr` caching** (CLAUDE.md Rule 6). The harness shifts input pointers by 256 B every iteration -- any data_ptr-keyed memoisation goes stale and fails correctness.

## Linked problems

- `.claude/knowledge/problems/problem_004_gemm_n128_k2048.md` -- L1, single-op GEMM, BF16 inputs.

Additional `problems/` dossiers covering the broader GEMM family (`problem_007_gemm_n4096_k4096`, `problem_011_gemm_n28672_k4096`, etc.) link back to this pattern via the matching-pattern field of their dossiers.
