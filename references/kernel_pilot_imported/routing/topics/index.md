# Kernel Topics Index (trimmed)

A short cross-reference of all topic pages. During a solbench loop, open the
topic page that matches the target operator and use it to choose code-first
sources to inspect.

| Topic | Page | Applies to (kept frameworks only) |
| --- | --- | --- |
| Attention / FMHA / Paged | `attention.md` | cutlass, flash-attention, flashinfer, pytorch, triton |
| GEMM / Tensor-Core matmul | `matmul-gemm.md` | cutlass, pytorch, triton |
| Mixture of Experts | `moe.md` | triton |
| Normalization (RMS / LN / QK-Norm) | `normalization.md` | pytorch, triton |
| RoPE | `rope.md` | flashinfer, triton |
| Activation / element-wise fusion | `activation-fusion.md` | pytorch, triton |
| Sampling / speculative decode | `sampling.md` | flashinfer |
| Quantization (FP8 / FP4 / INT8 / AWQ / GPTQ) | `quantization-fp8.md` | cutlass |
| KV-cache / paged memory | `kv-cache.md` | flashinfer |

`communication.md` was dropped from the trimmed import (solbench has 0 multi-GPU work).
Per-topic framework lists have been pruned to only the 5 kept frameworks (cutlass, flash-attention, flashinfer, pytorch, triton). See `KB_IMPORT_README.md` for full trim rationale.
