# Triton

> **SOLBENCH NOTICE.** This file was imported from kernel-pilot. solbench's
> submission surface is **CUDA C++ / CUTLASS C++ / CuTe C++ ONLY** (`CLAUDE.md` Rule 5).
> Triton / CuTe DSL Python / cuTile Python / `cudnn_frontend` Python / `torch.compile` /
> TileLang / ThunderKittens / QuACK referenced below are **algorithm references only** —
> lift the algorithm into CUDA C++ / CUTLASS C++ for the *.cu hand-port. solbench is
> also counter-free (no NCU per Rule 1); counter-free profiling lives in `scripts/baselines_harness/`.

Repository: <https://github.com/triton-lang/triton>

Deep reference: `references/source-guides/triton.md`

Triton is the dominant DSL for **portable, autotuned LLM kernels**. SGLang,
vLLM, and PyTorch Inductor all use Triton for fused MoE, fused norms,
attention variants, and elementwise fusion.

## Where the kernels live

| Directory | What you find there |
| --- | --- |
| `python/tutorials/` | Canonical Triton tutorials (matmul, softmax, attention, layer-norm). |
| `python/triton/language/` | Triton language primitives (`tl.dot`, `tl.load`, `tl.store`, `tl.atomic`, block pointers, TMA). |
| `python/triton/runtime/` | Autotuner, JIT cache, `do_bench`. |
| `lib/Conversion/` | MLIR conversion passes (informative for "why is my kernel slow on this arch"). |
| `test/TritonGPU/` | IR-level tests; reference for what the compiler optimizes. |

## Optimization patterns documented here

- **Block pointers + TMA**: the modern way to load tiles on Hopper / Blackwell
  Triton kernels.
- **Autotune configs**: `triton.Config({...}, num_warps=, num_stages=)`. The
  default Inductor configs are a strong starting point.
- **Software pipelining via `num_stages`**: increasing stages overlaps loads
  with compute, but increases smem pressure.
- **Persistent kernels**: a small grid that loops over tiles, used for
  Stream-K-like GEMMs and tail-aware MoE.
- **`tl.atomic_*` collectives**: the standard way to write fused-add
  outputs and reductions across program ids.

## Common pitfalls

- The same source produces different SASS on different GPU arches; always
  recompile and re-tune when changing arch.
- `tl.load(mask=)` with out-of-bounds addresses is cheap on H100 but
  *expensive* on A100; choose the mask strategy per arch.
- The autotuner caches by `key=` arguments only; missing a runtime-varying
  argument from `key=` causes the autotuner to pick a stale config.
- `num_stages` higher than 2 often regresses on smem-bound kernels.

## When to read this framework

- You are designing any Triton kernel.
- You are reading SGLang / vLLM / Inductor Triton kernels and want to
  understand the underlying primitives.

## Reuse / Copy Rules — SOLBENCH-CONSTRAINED

- **Do NOT copy Triton source directly into a solbench submission.** Submissions are CUDA C++ / CUTLASS C++ / CuTe C++ only (Rule 5). Triton tutorial kernels and downstream Triton baselines are **algorithm references**; lift the algorithm (tile shape, dtype handling, autotuner heuristic) into a CUDA C++ / CUTLASS C++ implementation.
- Record source path or URL, commit, the algorithm you lifted, and your CUDA C++ rewrite in `baselines/<category>/<slug>/README.md` before mutating.

