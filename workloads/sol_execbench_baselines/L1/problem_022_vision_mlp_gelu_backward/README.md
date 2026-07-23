# problem_022_vision_mlp_gelu_backward

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | fp32 Vision MLP + GELU-tanh backward pass. Captured from the Qwen3-VL-8B-Instruct vision encoder block. `hidden_size = 1280` and `intermediate_size = 3584` pinned. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_022_vision_mlp_gelu_backward.md`       |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/022_vision_mlp_gelu_backward/`                             |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Seven-kernel chain in fp32 CUDA C++ + CUTLASS Sm80 TensorOp.
Four fp32 GEMMs back-to-back with one per-element GELU-tanh
backward kernel between the two halves, plus two per-column
bias-sum reductions over `seq_len`.

The reference algorithm is a faithful backward of
`output = fc2( gelu( fc1(hidden_state) ) )` where GELU uses
the tanh approximation:

    grad_fc2_bias    = grad_output.sum(dim=0)
    grad_fc2_weight  = grad_output.t() @ gelu_output
    grad_gelu_output = grad_output       @ fc2_weight
    gelu_grad        = 0.5*(1+tanh(arg))
                     + 0.5*x*sech^2(arg)*d_arg_dx
                     where arg      = sqrt(2/pi) * (x + 0.044715 * x^3)
                           d_arg_dx = sqrt(2/pi) * (1 + 3*0.044715 * x^2)
                           x        = fc1_output
    grad_fc1_output  = grad_gelu_output * gelu_grad
    grad_fc1_bias    = grad_fc1_output.sum(dim=0)
    grad_fc1_weight  = grad_fc1_output.t() @ hidden_state
    grad_hidden_state= grad_fc1_output     @ fc1_weight

### GEMM-A — `grad_gelu_output = grad_output @ fc2_weight`

CUTLASS Sm80 TensorOp fp32 GEMM with mma.m16n8k8 tf32 mma:
- `M = seq_len`, `N = intermediate_size = 3584`,
  `K = hidden_size = 1280`.
- `ThreadblockShape<128, 128, 16>`, `WarpShape<64, 64, 16>`,
  `InstructionShape<16, 8, 8>`, `NumStages = 3`.
- Default `LinearCombination<float, 4, float, float>`
  epilogue (alpha = 1, beta = 0; no accumulator on C).

### GEMM-B — `grad_fc2_weight = grad_output.t() @ gelu_output`

CUTLASS Sm80 TensorOp fp32 GEMM with the same template
shape; A is read with `ColumnMajor` layout over the
underlying row-major `grad_output` storage (which
transposes the logical view):
- `M = hidden_size = 1280`, `N = intermediate_size = 3584`,
  `K = seq_len`.

### Bias-sum kernel — `grad_fc2_bias = grad_output.sum(dim=0)`

Per-(col) CTA over Grid `(hidden_size = 1280)`,
`BLOCK_THREADS = 256`. Each CTA stride-loops over `seq_len`
accumulating a per-lane partial; warp-shuffle reduces
within each warp, an 8-element shared-memory tree reduces
across the 8 warps. Lane 0 writes `grad_fc2_bias[col]`.

### GELU-tanh backward per-element kernel

One thread per `(i, j) ∈ [seq_len, intermediate_size]`. Each
thread loads `fc1_output[i, j]` and the matching
`grad_gelu_output[i, j]`, computes the GELU-tanh derivative,
and writes `grad_fc1_output[i, j]`. No reductions; pure
streaming.

### GEMM-C — `grad_hidden_state = grad_fc1_output @ fc1_weight`

Same Sm80 TensorOp template as GEMM-A. `M = seq_len`,
`N = hidden_size = 1280`,
`K = intermediate_size = 3584`. `fc1_weight` is stored as
`[intermediate_size, hidden_size]` row-major — the logical
`[K = intermediate_size, N = hidden_size]` view matches the
storage layout, so `LayoutInputB = RowMajor`.

### GEMM-D — `grad_fc1_weight = grad_fc1_output.t() @ hidden_state`

Same Sm80 TensorOp template as GEMM-B (A read as
`ColumnMajor` over row-major storage to encode the
transpose). `M = intermediate_size = 3584`,
`N = hidden_size = 1280`, `K = seq_len`.

### Bias-sum kernel — `grad_fc1_bias = grad_fc1_output.sum(dim=0)`

Same structure as the `grad_fc2_bias` kernel but with Grid
`(intermediate_size = 3584)`.

All per-column bias-sum reductions are in-CTA only; no
cross-CTA reduction tree means no accumulation-order drift
against PyTorch's reduction order — matches the safe per-
row reduction pattern used in the cohort's other LN /
softmax kernels.

### Allocations

`grad_gelu_output[S, intermediate_size]` and
`grad_fc1_output[S, intermediate_size]` are launcher-
allocated `torch::empty` workspaces. The five output
tensors (`grad_hidden_state`, `grad_fc1_weight`,
`grad_fc1_bias`, `grad_fc2_weight`, `grad_fc2_bias`) are
pre-allocated by the eval driver and written in-place
(DPS).

### Runtime-check audit

Every const axis from `definition.json` enforced on every
input and output tensor that the kernel indexes with that
fixed assumption:

| Tensor              | Runtime check                                              |
|---------------------|------------------------------------------------------------|
| `grad_output`       | `is_cuda`, `kFloat32`, `is_contiguous`, `dim == 2`, `size(1) == hidden_size = 1280` |
| `hidden_state`      | same dtype/contig, `dim == 2`, `size(0) == grad_output.size(0)`, `size(1) == hidden_size = 1280` |
| `fc1_weight`        | same dtype/contig, `dim == 2`, `size(0) == intermediate_size = 3584`, `size(1) == hidden_size = 1280` |
| `fc1_bias`          | same dtype/contig, `dim == 1`, `size(0) == intermediate_size = 3584` |
| `fc2_weight`        | same dtype/contig, `dim == 2`, `size(0) == hidden_size = 1280`, `size(1) == intermediate_size = 3584` |
| `fc2_bias`          | same dtype/contig, `dim == 1`, `size(0) == hidden_size = 1280` |
| `fc1_output`        | same dtype/contig, `dim == 2`, `size(0) == grad_output.size(0)`, `size(1) == intermediate_size = 3584` |
| `gelu_output`       | same dtype/contig, `dim == 2`, `size(0) == grad_output.size(0)`, `size(1) == intermediate_size = 3584` |
| `grad_hidden_state` | same dtype/contig, `dim == 2`, `sizes() == hidden_state.sizes()` |
| `grad_fc1_weight`   | same dtype/contig, `dim == 2`, `sizes() == fc1_weight.sizes()` |
| `grad_fc1_bias`     | same dtype/contig, `dim == 1`, `size(0) == intermediate_size = 3584` |
| `grad_fc2_weight`   | same dtype/contig, `dim == 2`, `sizes() == fc2_weight.sizes()` |
| `grad_fc2_bias`     | same dtype/contig, `dim == 1`, `size(0) == hidden_size = 1280` |

`fc1_bias` and `fc2_bias` are accepted in the entry signature
to match the reference's positional argument list but are not
read by the run — the backward pass through a `Linear(b)`
layer does not consume the forward `b`.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was
compile-checked and natively-correctness-validated on. Final
SOL scoring is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060):
  the larger backward FFN weight tensors (`fc1_weight` =
  17.5 MiB fp32, `fc2_weight` = 17.5 MiB fp32) substantially
  exceed the 3090 Ti's 6 MiB L2 partition. B200's HBM3e +
  larger L2 reshape the cross-CTA reuse pattern.
- DSMEM and cluster-cooperative tile staging (sm_100a-only)
  are out of scope at this Sm80-portability level.
- TCGen05 / TMEM divergence: all four GEMMs run on the
  Sm80 TensorOp path with tf32 mma, so B200's TCGen05 /
  TMEM is NOT exercised here.
- L2 carveout / partition: workload-dependent. Per-launch
  read traffic is approximately
  `(2 + 2) * S * hidden_size * 4 + 2 * S * intermediate_size * 4 +
  hidden_size * intermediate_size * 4 * 2 ≈ 35 MiB + 16 KiB * S`
  fp32 — well over the 3090 Ti's 6 MiB L2 at typical
  `S = 1024`.
- The per-column bias-sum kernels are bandwidth-bound on
  the column-major stride pattern of the row-major
  `grad_output` / `grad_fc1_output` tensors. The
  `seq_len`-axis stride is `hidden_size` (resp.
  `intermediate_size`), which produces
  strided global loads that the Sm80 L1 cache absorbs via
  its 128-byte sectors only partially.
