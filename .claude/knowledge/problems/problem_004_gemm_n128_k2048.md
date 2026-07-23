---
Use when: starting the L1 single-op GEMM baseline for `problem_004_gemm_n128_k2048` -- `C = A @ B.T` with `A: [M, K=2048] BF16`, `B: [N=128, K=2048] BF16`, `C: [M, N=128] BF16`. M varies across ~25 workloads from 1 to 16,294. Captured from Qwen 3 30B A3B MoE gate (`moe.gate`).
Primary raw source: references/NVIDIA_Manuals/solbench_problems/problem_004_gemm_n128_k2048.md
Supporting sources: .claude/knowledge/patterns/gemm_baseline.md; .claude/knowledge/sources/cutlass-cute-docs.md
Repo touchpoints: baselines/L1/problem_004_gemm_n128_k2048/cutlass/; cutlass/examples/cute/tutorial/; sol-execbench/examples/cutlass/gemm/
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
Version family: sol-execbench-2026
Default skill: implementing-sol-baselines
Routing tags: gemm; matmul; problem_004; bf16; qwen3; moe-gate
---

# problem_004_gemm_n128_k2048

## Use when

Implementing the baseline for the canonical `C = A @ B.T` GEMM at N=128, K=2048, dtype BF16. This is a small-N "skinny" matmul characteristic of MoE gate projections.

## Workload summary

| Field | Value |
|---|---|
| Operation | `C = A @ B.T` (single matmul) |
| Dtypes | BF16 inputs, BF16 output; FP32 accumulator required per CLAUDE.md Rule 8 |
| Shapes | `A: [M, 2048]`, `B: [128, 2048]`, `C: [M, 128]` |
| M range | 1 to 16,294 across ~25 workloads |
| Source model | Qwen 3 30B A3B MoE gate (`moe.gate`) |
| Category | L1 (single-op) |

## Matching pattern

This problem instantiates `.claude/knowledge/patterns/gemm_baseline.md` with `(N, K) = (128, 2048)`. The pattern card carries the canonical CUTLASS recipe (CollectiveMainloop + TiledMma + EpilogueVisitor + KernelTmaWarpSpecializedCooperative); this dossier records the problem-specific constraints.

## Implementation entry point

Per CLAUDE.md Rule 5 the submission surface is CUTLASS C++ or CUDA C++. Author the baseline under:

- `baselines/L1/problem_004_gemm_n128_k2048/cutlass/{main.cpp, kernel.cu, kernel.h, solution.json}` for the CUTLASS variant.
- `baselines/L1/problem_004_gemm_n128_k2048/cuda_cpp/{kernel.cu, solution.json}` for a from-scratch CUDA variant (optional).

Entry point signature per `sol-execbench/docs/solution.md`:

```cpp
void run(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    // launch GEMM, write result into C in-place; no return
}
```

The eval driver pre-allocates `C` and passes it in DPS style.

## Shape-dependent considerations

- **Small M (M <= 64).** Heavily memory-bound. The matrix `B` (128 x 2048 x 2 bytes = 512 KB BF16) fits comfortably in B200 L2 and is reused across small-M batches. Prefer a sm_100a schedule that keeps `B` in L2 across the launch.
- **Mid M (M ~ 100 - 1000).** Mixed bound. Tile shapes like 64x128x64 or 128x128x64 reasonable.
- **Large M (M ~ 8000 - 16000).** Compute-bound. Use the largest tile (256x128x64 or 128x128x128) the cluster supports.

## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline

The 5060 (sm_120) can compile and run this baseline for correctness verification; final scoring is on B200 (sm_100a). Concrete divergences:

- The TCGen05 / TMEM-accumulating schedule available on sm_100a is unavailable on sm_120; the collective builder will pick `KernelMultistage` automatically.
- sm_120's L2 cannot hold `B` across all batches the way B200's much-larger L2 does -- the local timing will under-represent the B200 cache-reuse win.
- BF16 tensor-core latency is similar between arches, but cooperative-cluster wins (cluster size > 1) are sm_100a-only for this small-N GEMM.

Evidence-label vocabulary per `CLAUDE.md` lives in `baselines/L1/problem_004_gemm_n128_k2048/README.md`, NOT in this dossier.

## Cross-references

- Pattern recipe: `.claude/knowledge/patterns/gemm_baseline.md`.
- Source spec (authoritative): `references/NVIDIA_Manuals/solbench_problems/problem_004_gemm_n128_k2048.md`.
- CUTLASS reference: `sol-execbench/examples/cutlass/gemm/`.
