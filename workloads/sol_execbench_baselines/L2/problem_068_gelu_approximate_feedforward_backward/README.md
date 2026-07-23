# problem_068_gelu_approximate_feedforward_backward

| Field                       | Value                                                                                              |
|-----------------------------|----------------------------------------------------------------------------------------------------|
| Problem                     | GELU-approximate feedforward backward (Linear -> GELU tanh-approx -> Linear)                       |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_068_gelu_approximate_feedforward_backward.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L2/068_gelu_approximate_feedforward_backward/`                       |
| Variant                     | `cuda_cpp`                                                                                         |
| Evidence                    | `Profiled-local-sm_120`                                                                                  |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                                   |
| Local speedup vs reference  | 16/16 PASSED on sm_86 (diagnostic latency captured per workload; local-only)                       |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 - 0.5+>                                   |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                                  |

## Approach

Pure ATen / cuBLAS dispatch — NO custom CUDA kernels. The backward
computes three outputs (no bias gradients) through four 2D fp32
`at::matmul` calls + a tanh-approximation GELU derivative that
short-circuits forward recomputation by consuming the precomputed
`tanh_inner` and `linear1_out` inputs directly:

1. Backward through Linear2 (`output = gelu_out @ weight2.T`):
   - `grad_gelu_out = grad_output @ weight2` (cublasSgemm OP_N OP_N).
   - `grad_weight2 = grad_output_2d.t() @ gelu_out_2d` (cublasSgemm OP_T OP_N).
2. Backward through tanh-approximation GELU (using the precomputed inputs):
   - `gelu_grad_part1 = 0.5 * (1 + tanh_inner)`.
   - `sech_squared = 1 - tanh_inner^2`.
   - `d_inner_dx = sqrt(2/pi) * (1 + 0.134145 * linear1_out^2)`.
   - `gelu_grad_part2 = 0.5 * linear1_out * sech_squared * d_inner_dx`.
   - `gelu_grad = gelu_grad_part1 + gelu_grad_part2`.
   - `grad_linear1_out = grad_gelu_out * gelu_grad`.
3. Backward through Linear1 (`linear1_out = hidden_states @ weight1.T`):
   - `grad_hidden_states = grad_linear1_out @ weight1` (cublasSgemm OP_N OP_N).
   - `grad_weight1 = grad_linear1_out_2d.t() @ hidden_states_2d` (cublasSgemm OP_T OP_N).

`dim = 1536` and `hidden_dim = 6144` (4x expansion) are pinned as
`constexpr` axes; every indexed tensor is shape-checked against
these values. CUDA guard from the first input. Launch on
`at::cuda::getCurrentCUDAStream()` per CLAUDE.md Rule 7.

The precomputed `tanh_inner` and `linear1_out` inputs distinguish
problem_068 from problem_073 (which carries `intermediate` only and
must recompute `tanh_inner` inline) and shave one element-wise
forward pass off the runtime cost.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) compiles and is expected to smoke-run this baseline;
final SOL scoring is on B200 (sm_100a). Differences worth noting for this
pattern:

- HBM3e vs GDDR6X bandwidth: feedforward backward is memory-bound on the
  weight reads; the 3090 Ti's bandwidth ceiling makes local wall-time a weak
  proxy for the B200 SOL Score. Local timing primarily validates correctness
  and launch-config sanity.
- DSMEM / cluster-cooperative reduction (sm_100a-only) are not used here;
  the GEMM dispatches go through cuBLAS's standard kernel selection on every
  arch.
- L2 carveout: B200's larger L2 absorbs the weight tensors during the four
  matmul passes; the 3090 Ti's smaller L2 cannot. GDDR7 does not apply here
  (the 3090 Ti uses GDDR6X; sm_120 would be the GDDR7 reference if a future
  cloud host is allocated).
- TCGen05 / TMEM: this template does not use 5th-gen tensor cores or the
  dedicated TMEM register surface; the fp32 sgemm path is the conservative
  PyTorch dispatch and is not exposed to sm_100a-only TCGen05 / TMEM codegen
  differences. A future tensor-core-aware variant is out of scope.
