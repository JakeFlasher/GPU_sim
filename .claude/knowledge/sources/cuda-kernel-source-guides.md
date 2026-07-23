---
Use when: you need a code map for one of the 5 upstream frameworks kept after the trim: CUTLASS, FlashAttention, FlashInfer, PyTorch, Triton (Triton is REFERENCE ONLY — Rule 5 forbids submission). Use to find where the kernel directories, key classes / functions / file paths, tests, and benchmarks live in each upstream repo before lifting an algorithm into a *.cu hand-port. 13 other framework source-guides were dropped in the post-pivot trim.
Primary raw source: references/kernel_pilot_imported/references/source-guides/
Supporting sources: references/kernel_pilot_imported/references/index.md; references/kernel_pilot_imported/KB_IMPORT_README.md
Repo touchpoints: references/kernel_pilot_imported/references/source-guides/
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
Version family: kernel-pilot-source-guides-trimmed-2026-05
Default skill: implementing-sol-baselines
Routing tags: kernel-pilot; source-guides; code-maps; cutlass; flash-attention; flashinfer; pytorch; triton
---

# CUDA Kernel Source Guides (Imported, Trimmed)

## Use when

You need to find a kernel implementation in one of 5 kept upstream repos, identify the callable wrappers / tests / benchmarks, or check the file layout before lifting code into a *.cu hand-port.

## Approach / Content

| Guide | Repo | Scope |
|---|---|---|
| `cutlass.md` | NVIDIA/cutlass | `include/`, `examples/`, `test/` — **submission target** |
| `flash-attention.md` | Dao-AILab/flash-attention | `flash_attn/flash_attn_interface.py` + csrc — algorithm reference |
| `flashinfer.md` | flashinfer-ai/flashinfer | `csrc/` — algorithm reference for FIB category |
| `pytorch.md` | pytorch/pytorch | `aten/src/ATen/native/cuda/` — reference baseline binding |
| `triton.md` | triton-lang/triton | `python/triton/`, `lib/Dialect/Triton/` — **REFERENCE ONLY (Rule 5)** |

Each guide also lists:
- Key callable wrappers / Python bindings
- Test entry points
- Benchmark entry points
- Relevant CMake / build directories

**Dropped from this layer:** tensorrt-llm, cccl-cub, thunderkittens, tilelang, quack, tilekernels, veitner-blog, colfax-research, cuda-blog-kernels, deepgemm, sglang, vllm, cute-dsl. For these, query GitHub directly.

## Architecture portability

Many upstream kernels are Hopper-tuned (sm_90) or Ada-tuned (sm_89). When lifting into a B200 (sm_100a) hand-port, re-derive: TMA descriptors, async-copy semantics, cluster behavior, DSMEM, register/occupancy, L2 partitioning, NVFP4 block-scaling layout, FP8 e4m3/e5m2 layout, CUTLASS schedule selection. Solbench's local proxy at sm_120 (RTX 5060) only compile-checks the path; it does NOT validate Blackwell behavior.

## Cross-references

- `references/kernel_pilot_imported/references/prs/cutlass.md` for one production CUTLASS PR evidence example (the rest of the PR corpus was trimmed).
- Solbench's `references/NVIDIA_Manuals/cutlass_cute/` (717 files) is the authoritative CUTLASS / CuTe C++ doc mirror — prefer it for primitive-level questions.
