# problem_004_attention_output_projection_with_reshape_backward

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | bf16 backward pass for the attention output projection with shape conversion (transpose + reshape + linear in the forward; reverse in the backward). `num_heads = 32`, `head_dim = 64`, `hidden_size = 2048` (= num_heads × head_dim) pinned. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_004_attention_output_projection_with_reshape_backward.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/004_attention_output_projection_with_reshape_backward/`    |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Pure ATen/cuBLAS dispatch — NO custom CUDA kernels. The reference
algorithm decomposes into two fp32 GEMMs (after a bf16 → fp32
upcast) plus a reshape + transpose, with the final outputs cast
back to bf16:

    grad_output_f32  = grad_output.to(fp32)
    reshaped_f32     = reshaped.to(fp32)
    weight_f32       = weight.to(fp32)
    grad_output_2d   = grad_output_f32.reshape(-1, hidden_size)
    reshaped_2d      = reshaped_f32.reshape(-1, hidden_size)
    grad_weight_f32  = grad_output_2d.t() @ reshaped_2d
    grad_reshaped_2d = grad_output_2d @ weight_f32
    grad_reshaped    = grad_reshaped_2d.reshape(B, S, hidden_size)
    grad_transposed  = grad_reshaped.reshape(B, S, num_heads, head_dim)
    grad_attn_output = grad_transposed.transpose(1, 2).contiguous()
    output bf16: grad_attn_output.to(bf16), grad_weight.to(bf16)

For fp32 inputs on Sm80, PyTorch's `at::mm` dispatches
`cublasSgemm` via the cuBLAS handle; the deployed baseline's
two `at::mm` calls (GEMM A: `grad_output.t() @ reshaped`;
GEMM B: `grad_output @ weight_f32`) route through the same
cuBLAS algorithm-selection path the reference uses.

Implementation in `main.cpp`:

1. **Upcast bf16 → fp32** for the three input tensors via
   `at::to(torch::kFloat32)`. These are workspace
   allocations.
2. **Reshape** `grad_output_f32` and `reshaped_f32` to
   2D `(B*S, hidden_size)`.
3. **fp32 cuBLAS sgemm A**:
   `grad_weight_f32 = grad_output_2d.t().mm(reshaped_2d)` →
   `(hidden_size, hidden_size)`. M=2048, K=B*S, N=2048.
4. **Downcast → bf16** and write into the pre-allocated DPS
   `grad_weight` via `.copy_(grad_weight_f32.to(bf16))`.
5. **fp32 cuBLAS sgemm B**:
   `grad_reshaped_2d = grad_output_2d.mm(weight_f32)` →
   `(B*S, hidden_size)`. M=B*S, K=2048, N=2048.
6. **Reshape + transpose** `(B*S, hidden_size)` →
   `(B, S, num_heads, head_dim)` →
   `(B, num_heads, S, head_dim)` via ATen
   `.reshape(...).transpose(1, 2).contiguous()`.
7. **Downcast → bf16** and write into the pre-allocated DPS
   `grad_attn_output` via `.copy_(...to(bf16))`.

### Why fp32 inner compute (not bf16)

The reference upcasts all bf16 inputs to fp32 before the
GEMMs, then downcasts the outputs back to bf16. A
bf16-in-bf16-out GEMM would produce a different
intermediate precision than the reference's fp32-inner
chain, and would likely violate the workload's correctness
tolerance band. The deployed baseline matches the
reference's dtype contract exactly.

### Allocations

All workspaces (`grad_output_f32`, `reshaped_f32`,
`weight_f32`, `grad_output_2d`, `reshaped_2d`,
`grad_weight_f32`, `grad_reshaped_2d`, `grad_reshaped`,
`grad_attn_output_f32`) are allocated implicitly by ATen.
The two DPS outputs (`grad_attn_output` and `grad_weight`)
are pre-allocated by the eval driver and written in-place
via `.copy_(...to(bf16))`.

A dummy `kernel.cu` (with just
`#include <cuda_runtime.h>`) is included to force
`torch.utils.cpp_extension.load` into its CUDA build path,
since `main.cpp`'s ATen headers transitively require
`cuda_runtime_api.h`. Same workaround as
`problem_065_fused_statistics_projection_and_split`.

### Runtime-check audit

Every const axis from `definition.json` enforced on every
input and output tensor that the kernel indexes with that
fixed assumption:

| Tensor             | Runtime check                                              |
|--------------------|------------------------------------------------------------|
| `grad_output`      | `is_cuda`, `kBFloat16`, `is_contiguous`, `dim == 3`, `size(2) == hidden_size = 2048` |
| `reshaped`         | same dtype/contig, `dim == 3`, `size(0) == grad_output.size(0)`, `size(1) == grad_output.size(1)`, `size(2) == hidden_size = 2048` |
| `weight`           | same dtype/contig, `dim == 2`, `size(0) == hidden_size = 2048`, `size(1) == hidden_size = 2048` |
| `grad_attn_output` | same dtype/contig, `dim == 4`, `size(0) == B`, `size(1) == num_heads = 32`, `size(2) == S`, `size(3) == head_dim = 64` |
| `grad_weight`      | same dtype/contig, `dim == 2`, `size(0) == hidden_size = 2048`, `size(1) == hidden_size = 2048` |

Destination-passing-style entry: both DPS outputs are
pre-allocated. Each is written exactly once via
`.copy_(...to(bf16))`.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was
compile-checked and natively-correctness-validated on. Final
SOL scoring is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060):
  the two fp32 sgemm calls operate on `[hidden_size,
  hidden_size]` weight tile (16 MiB fp32) — well over the
  3090 Ti's 6 MiB L2 partition. B200's larger L2 will
  capture more mainloop reuse; cuBLAS may also select a
  different sgemm kernel.
- DSMEM and cluster-cooperative tile staging (sm_100a-only)
  are out of scope at this Sm80-portability level.
- TCGen05 / TMEM divergence: the two fp32 sgemm calls run
  on Ampere's tf32-mma FP32 SM pipe; B200's TCGen05 /
  TMEM is NOT exercised at this dtype.
- L2 carveout / partition: workload-dependent. The bf16 →
  fp32 upcasts double the read traffic relative to a pure
  bf16 path, but the fp32 GEMMs have stronger numerical
  fidelity that matches the reference's tolerance band.
- The deployed pure-ATen/cuBLAS path means the cohort-B
  probe (cublasSgemm at the equivalent shape) measures the
  same algorithm-selection path the deployed code uses —
  probe runtime-contract fidelity per BL-20260515 is
  preserved at the kernel-selection layer.
