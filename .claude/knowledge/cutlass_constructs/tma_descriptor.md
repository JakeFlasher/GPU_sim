---
Use when: building or consuming a TMA (Tensor Memory Accelerator) descriptor for async global<->shared copies on sm_100a. Used by every CUTLASS / CuTe kernel that wants single-instruction bulk transfer plus implicit prefetch + masking. The `cp.async.bulk.tensor` and `cp.async.bulk.tensor.gpu_to_smem` PTX ops are the underlying instructions.
Primary raw source: cutlass/include/cute/atom/copy_atom.hpp
Supporting sources: cutlass/include/cute/algorithm/copy.hpp; references/NVIDIA_Manuals/cutlass_cute/; references/NVIDIA_Manuals/ptx_isa/
Repo touchpoints: cutlass/include/cute/atom/copy_atom.hpp; cutlass/include/cute/atom/copy_traits.hpp
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
Routing tags: tma; tma-descriptor; cp-async-bulk; copy-atom; bulk-tensor; gemm; attention
---

# TMA Descriptor (CUTLASS Construct)

## Use when

Setting up a TMA-mediated async copy from global memory into shared memory. The descriptor encodes the global tensor's base pointer, shape, stride, swizzle policy, and the per-CTA tile that each `cp.async.bulk.tensor` issue retrieves.

## Canonical recipe (host-side descriptor build via CuTe)

```cpp
auto tma_load = cute::make_tma_copy(
    cute::SM100_TMA_LOAD_MULTICAST{},        // sm_100a multicast variant
    tensor_global,                            // CuTe global Tensor
    Layout<Shape, Stride>{},                  // shared-memory destination layout
    cute::Tile<_128, _64, _32>{}              // per-issue tile shape
);
```

Then on device:

```cpp
// In producer warp:
auto tma_thr = tma_load.get_slice(producer_warp_idx);
cute::copy(tma_load, tma_thr.partition_S(gtensor), tma_thr.partition_D(stensor));
cute::cp_async_bulk_commit_group();
cute::cp_async_bulk_wait_group<N>();
```

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

TMA descriptors are sm_100a-only (with limited sm_90 / sm_120 backports). The CuTe TMA copy atom degrades on sm_120 to `cp.async.bulk` without indirect-descriptor support, then to scalar `cp.async` without bulk staging. Local timing on sm_120 does NOT reflect sm_100a's TMA throughput.

Evidence-label conventions per `CLAUDE.md` live in `baselines/<category>/<slug>/README.md`, NOT in this construct card.

## Pattern cards that compose this construct

`gemm_baseline`, `attention_core`, `attention_backward`, `flash_attention`, `gqa_paged_decode`, `gqa_paged_prefill`, `mla_paged_attention`, `moe_dispatch`, `rope_attention`, `transformer_layer`, `mlp_gelu_geglu`, `swiglu_mlp`.
