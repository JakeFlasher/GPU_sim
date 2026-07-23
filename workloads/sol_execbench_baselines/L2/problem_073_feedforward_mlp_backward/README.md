# problem_073_feedforward_mlp_backward

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | Feedforward MLP backward (Linear -> GELU tanh-approx -> Linear)                             |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_073_feedforward_mlp_backward.md`       |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L2/073_feedforward_mlp_backward/`                             |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Compile-checked-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                             |
| Local speedup vs reference  | 0.82 - 1.27x on sm_86 across 16/16 PASSED workloads (diagnostic only; local-only)           |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 - 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Pure ATen / cuBLAS dispatch — NO custom CUDA kernels. The backward
computes seven outputs through four 2D fp32 `at::matmul` calls + a
tanh-approximation GELU derivative as elementwise arithmetic + two
batch/seq summations:

1. Backward through Linear2 (`output = intermediate_activated @ weight2.T + bias2`):
   - `grad_intermediate_activated = grad_output @ weight2`
     (`at::matmul` with both operands in OP_N orientation -> cublasSgemm OP_N OP_N).
   - `grad_weight2 = grad_output_2d.t() @ intermediate_activated_2d`
     (cublasSgemm OP_T OP_N).
   - `grad_bias2 = grad_output.sum({0, 1})`.
2. Backward through tanh-approximation GELU:
   - `inner = sqrt(2/pi) * (x + 0.044715 * x^3)`, `tanh_inner = tanh(inner)`,
     `d_inner_dx = sqrt(2/pi) * (1 + 0.134145 * x^2)`,
     `sech_squared = 1 - tanh_inner^2`,
     `gelu_grad = 0.5 * (1 + tanh_inner) + 0.5 * x * sech_squared * d_inner_dx`.
   - `grad_intermediate = grad_intermediate_activated * gelu_grad`.
3. Backward through Linear1 (`intermediate = hidden_states @ weight1.T + bias1`):
   - `grad_hidden_states = grad_intermediate @ weight1` (OP_N OP_N).
   - `grad_weight1 = grad_intermediate_2d.t() @ hidden_states_2d` (OP_T OP_N).
   - `grad_bias1 = grad_intermediate.sum({0, 1})`.

`hidden_size = 2048` and `intermediate_size = 8192` are pinned as
`constexpr` axes; every indexed tensor is shape-checked against
these values. CUDA guard from the first input. Launch on
`at::cuda::getCurrentCUDAStream()` per CLAUDE.md Rule 7.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) compiles and is expected to smoke-run this baseline;
final SOL scoring is on B200 (sm_100a). Differences worth noting for this
pattern:

- HBM3e vs GDDR6X bandwidth: MLP backward is memory-bound on the weight reads;
  the 3090 Ti's bandwidth ceiling makes local wall-time a weak proxy for the
  B200 SOL Score. Local timing primarily validates correctness and
  launch-config sanity.
- DSMEM / cluster-cooperative reduction (sm_100a-only) are not used here;
  the GEMM dispatches go through cuBLAS's standard kernel selection on every
  arch and do not exploit the cluster-cooperative SMs B200 offers.
- L2 carveout: B200's larger L2 absorbs the weight tensors during the four
  matmul passes; the 3090 Ti's smaller L2 cannot. GDDR7 does not apply here
  (the 3090 Ti uses GDDR6X; sm_120 would be the GDDR7 reference if a future
  cloud host is allocated).
- TCGen05 / TMEM: this template does not use 5th-gen tensor cores or the
  dedicated TMEM register surface; the fp32 sgemm path is the conservative
  PyTorch dispatch and is not exposed to sm_100a-only TCGen05 / TMEM codegen
  differences. A future tensor-core-aware variant of this kernel (which would
  touch TCGen05 / TMEM on B200) is out of scope for the current template.
