# PyTorch Kernel Reference

> **SOLBENCH NOTICE.** This file was imported from kernel-pilot. solbench's
> submission surface is **CUDA C++ / CUTLASS C++ / CuTe C++ ONLY** (`CLAUDE.md` Rule 5).
> Triton / CuTe DSL Python / cuTile Python / `cudnn_frontend` Python / `torch.compile` /
> TileLang / ThunderKittens / QuACK referenced below are **algorithm references only** —
> lift the algorithm into CUDA C++ / CUTLASS C++ for the *.cu hand-port. solbench is
> also counter-free (no NCU per Rule 1); counter-free profiling lives in `scripts/baselines_harness/`.

Repository: <https://github.com/pytorch/pytorch>

Source-only policy: do not query PyTorch PR notes. PyTorch is too large/noisy
for useful PR recall; inspect the source paths and current generated code
directly.

Use PyTorch as correctness oracle, semantic reference, broad-shape behavior
reference, and sometimes the candidate-code source when the baseline is ATen,
Inductor, Triton, or cuDNN integration code.

## Read Order

1. Python op schema and dispatch entry.
2. ATen native implementation or decomposition.
3. CUDA kernel and TensorIterator/reduction helpers.
4. Test coverage for semantics and edge cases.
5. Inductor or cuDNN path when the real baseline is `torch.compile` or cuDNN.

## Code Map

| Area | Paths to inspect |
| --- | --- |
| ATen native CUDA | `aten/src/ATen/native/cuda/` |
| TensorIterator | `aten/src/ATen/native/cuda/Loops.cuh`, TensorIterator helpers |
| Reductions | `aten/src/ATen/native/cuda/Reduce.cuh`, block/warp reduction helpers |
| Transformers / SDPA | `aten/src/ATen/native/transformers/cuda/` |
| cuDNN integration | `aten/src/ATen/native/cudnn/` |
| Inductor | `torch/_inductor/`, generated Triton templates and benchmarking utilities |
| Tests | `test/test_cuda.py`, `test/test_transformers.py`, op-specific tests |

## Search Patterns

```bash
rg -n "TORCH_IMPL_FUNC|REGISTER_DISPATCH|TensorIterator|gpu_kernel|Reduce|BlockReduce|sdpa|scaled_dot_product" aten torch test
rg -n "inductor|triton|do_bench|autotune|template" torch/_inductor test
rg -n "cuda|cudnn|flash|mem_efficient|math" aten/src/ATen/native/transformers
```

## Baseline Extraction

- Use PyTorch eager as semantic reference, not necessarily performance target.
- Record strides, broadcasting, type promotion, accumulation dtype, NaN/Inf
  behavior, rounding tolerance, and non-contiguous behavior.
- If comparing to `torch.compile`, save the generated kernel or at least the
  Inductor benchmark and config metadata.

## Candidate Translation

Translate:

- exact op semantics and edge cases
- output tolerance and dtype behavior
- representative shape generation
- decomposition of a composite op into tensors

If PyTorch/Inductor code is the requested starting point, copy/adapt it into the
standalone repo only with license/notice context, exact source path, commit, and
delta recorded. Otherwise translate the semantics and shape contracts above.

## Useful Cross-Framework Priors

- SGLang/vLLM for serving-specific narrowed shape contracts.
- CUTLASS/TensorRT-LLM for GEMM and quantized matmul prior art.
- FlashInfer/FlashAttention for attention-specific behavior.
