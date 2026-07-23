---
Use when: working with `tcgen05.mma` instructions and TMEM (Tensor Memory) accumulator residence — the sm_100a-only register file that holds MMA accumulators outside the regular register space. Critical for understanding why local sm_120 timing does not transfer to B200 scoring.
Primary raw source: cutlass/include/cutlass/detail/sm100_tmem_helper.hpp
Supporting sources: cutlass/include/cute/atom/mma_traits_sm100.hpp; references/NVIDIA_Manuals/blackwell_tuning/; references/NVIDIA_Manuals/ptx_isa/
Repo touchpoints: cutlass/include/cutlass/detail/sm100_tmem_helper.hpp; cutlass/include/cute/atom/mma_traits_sm100.hpp
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
Routing tags: tcgen05; tmem; tensor-memory; mma; gemm; attention; b200; blackwell
Also relevant skills: tuning-blackwell-sm120-kernels
---

# TCGen05 / TMEM (CUTLASS Construct)

## Use when

Reasoning about the sm_100a fifth-generation tensor-core instructions (`tcgen05.mma`, `tcgen05.cp`) and the TMEM accumulator register file they read from / write to.

## Concept

TMEM is a dedicated on-chip register file (separate from the regular per-thread register file) that holds MMA accumulators for the producer-consumer warp-specialised schedule. `tcgen05.mma` reads its A and B operands from shared memory (staged via TMA) and accumulates into TMEM; `tcgen05.cp` moves data between TMEM and shared memory / registers.

Compared to the Hopper-era `wgmma.mma_async`, TMEM-resident accumulation:
- Avoids the regular-register pressure that `wgmma` accumulator-tile sizing imposed.
- Enables larger output-tile shapes for the same warp count.
- Lets the epilogue chain pull accumulator slices via `tcgen05.cp` without spilling.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

TMEM and `tcgen05.*` are sm_100a-only. On sm_120 the CUTLASS collective builder degrades the schedule to `KernelMultistage` with accumulator-in-registers, paying register-pressure cost. This is one of the largest sm_120 -> sm_100a timing-translation gaps in the whole atlas; local sm_120 perf significantly under-represents the B200 score.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this construct card.

## Pattern cards that compose this construct

`gemm_baseline`, `attention_core`, `attention_backward`, `flash_attention`, `fp8_blockwise_quant`, `gqa_paged_decode`, `gqa_paged_prefill`, `mla_paged_attention`, `mlp_gelu_geglu`, `swiglu_mlp`, `moe_dispatch`, `transformer_layer`.
