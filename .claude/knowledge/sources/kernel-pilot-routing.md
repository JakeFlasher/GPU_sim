---
Use when: looking for a topic / framework routing layer across the trimmed kernel-pilot CUDA optimization knowledge pack at `references/kernel_pilot_imported/`. Use for the 5 kept frameworks (CUTLASS, FlashAttention, FlashInfer, PyTorch, Triton-as-algorithm-reference) and the 9 kept topics (attention, GEMM, MoE, normalization, RoPE, activation fusion, sampling, quantization-fp8, kv-cache). 13 unused frameworks and `communication` topic were dropped in the post-pivot trim.
Primary raw source: references/kernel_pilot_imported/routing/
Supporting sources: references/kernel_pilot_imported/KB_IMPORT_README.md; references/kernel_pilot_imported/README.md; references/kernel_pilot_imported/index.json; .claude/skills/kernel-knowledge/SKILL.md
Repo touchpoints: references/kernel_pilot_imported/routing/topics/; references/kernel_pilot_imported/routing/frameworks/; references/kernel_pilot_imported/index.json
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
Version family: kernel-pilot-trimmed-2026-05
Default skill: implementing-sol-baselines
Routing tags: kernel-pilot; routing; frameworks; topics; cutlass; flash-attention; flashinfer; pytorch; triton; attention; gemm; moe; normalization; rope; activation-fusion; sampling; quantization-fp8; kv-cache
---

# Kernel-Pilot Routing Layer (Imported, Trimmed)

## Use when

You need a topic-first or framework-first entry into the trimmed kernel-pilot knowledge pack. Read this stub, then jump to `references/kernel_pilot_imported/routing/topics/<topic>.md` or `references/kernel_pilot_imported/routing/frameworks/<framework>.md`.

## Approach / Content

**Topics** (`routing/topics/`, 9 topic pages + `index.md` = 10 files): `attention`, `matmul-gemm`, `moe`, `normalization`, `rope`, `activation-fusion`, `sampling`, `quantization-fp8`, `kv-cache`, plus `index.md`. `communication.md` was dropped (solbench has 0 multi-GPU work). These topic pages overlap with solbench's 27 `patterns/` cards — use the topic pages for cross-framework optimization vocabulary; use solbench's `patterns/<slug>.md` for the canonical CUTLASS / CuTe C++ recipe with arch-portability + forbidden-surface annotations.

**Frameworks** (`routing/frameworks/`, 5 files): `cutlass`, `flash-attention`, `flashinfer`, `pytorch`, `triton`. The 13 unused frameworks (`tensorrt-llm`, `cccl-cub`, `thunderkittens`, `tilelang`, `quack`, `tilekernels`, `veitner-blog`, `colfax-research`, `cuda-blog-kernels`, `deepgemm`, `sglang`, `vllm`, `cute-dsl`) were dropped in the post-pivot trim. For research on those, query GitHub directly.

Cross-references from the routing layer:

- Topic pages → `references/kernel_pilot_imported/references/source-guides/<repo>.md` for the kept frameworks.
- Framework pages → `references/kernel_pilot_imported/references/source-guides/<repo>.md`.
- For CUTLASS only: `references/kernel_pilot_imported/references/prs/cutlass.md` (one production CUTLASS PR evidence page).

## Architecture portability

`sm_100a` (B200) is solbench's scoring target. `sm_120` (RTX 5060) is solbench's local dev proxy only. The routing layer is architecture-agnostic — it points to source code. Architecture-specific guides are at `references/kernel_pilot_imported/references/ako4all/cuda-cpp/sm100-optimization-guide.md` and `cutlass-cpp/sm100-optimization-guide.md`. Solbench's own `references/NVIDIA_Manuals/blackwell_tuning/` is the authoritative Blackwell tuning source — prefer it when conflicts arise.
