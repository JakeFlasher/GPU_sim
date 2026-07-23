---
Use when: looking for one production CUTLASS PR evidence page in the trimmed kernel-pilot import. The broad PR research corpus (8 framework PR pages, 11 by-topic aggregations, pr-index.json, audit, open-watchlist) was dropped in the post-pivot trim; only `references/kernel_pilot_imported/references/prs/cutlass.md` is kept as a template / one-page production CUTLASS PR evidence example. For broader plateau research on production PRs, query GitHub directly or re-import the kernel-pilot PR layer.
Primary raw source: references/kernel_pilot_imported/references/prs/cutlass.md
Supporting sources: references/kernel_pilot_imported/KB_IMPORT_README.md; references/kernel_pilot_imported/references/index.md
Repo touchpoints: references/kernel_pilot_imported/references/prs/cutlass.md
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
Version family: kernel-pilot-prs-trimmed-2026-05
Default skill: implementing-sol-baselines
Routing tags: kernel-pilot; prs; pull-requests; optimization-prs; cutlass-prs; cutlass; gemm; tensor-cores
---

# CUDA Optimization PRs (Imported, Trimmed to CUTLASS-only)

## Use when

You need a production-PR optimization example for CUTLASS work. Use as the one kept entry from the original kernel-pilot PR corpus. For broader cross-framework PR research, query GitHub directly — the rest of the original PR layer was dropped in the post-pivot trim.

## Approach / Content

**The single kept page:** `references/kernel_pilot_imported/references/prs/cutlass.md` — production CUTLASS PRs (kernel changes, optimization recipes, changed paths).

**What was dropped during the trim:**

- 8 framework PR pages: `flash-attention.md`, `flashinfer.md`, `sglang.md`, `vllm.md`, `tensorrt-llm.md`, `tilelang.md`, `deepgemm.md`, `cccl-cub.md`
- 11 cross-repo by-topic aggregations under `prs/by-topic/`
- `prs/index.md`, `prs/audit.md`, `prs/open-watchlist.md`
- `prs/pr-index.json` (1.6 MB structured metadata)
- `prs/pr-scan-cache.json` (dropped earlier — raw NCU-bearing GitHub PR excerpts)

Drop rationale: solbench's active work (metric_extractor RLCR loop) doesn't depend on cross-framework PR research. None of the 11 implemented baselines cite an upstream framework PR as algorithm source. Re-import if solbench resumes broad baseline-porting research.

## Architecture portability

PR-level optimizations span Hopper (sm_90), Ada (sm_89), Ampere (sm_80/sm_86), and increasingly Blackwell (sm_100). When lifting an algorithm into a *.cu hand-port, re-derive tile shapes, TMA descriptors, cluster behavior, NVFP4/FP8 block-scaling layout for sm_100a (B200). Validate on solbench's five counter-free surfaces (NVBit / CUPTI Activity / Nsys / static introspection / compute-sanitizer).
