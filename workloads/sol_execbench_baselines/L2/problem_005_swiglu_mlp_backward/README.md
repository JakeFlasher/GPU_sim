# problem_005_swiglu_mlp_backward

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | Backward pass for the SwiGLU MLP `output = down_proj(silu(gate_proj(x)) * up_proj(x))`. Outputs grad_x, grad_gate_weight, grad_up_weight, grad_down_weight. hidden_size=16384, intermediate_size=53248 pinned. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_005_swiglu_mlp_backward.md`            |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L2/005_swiglu_mlp_backward/`                                  |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Compile-checked-sm_120`                                                                           |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Thirteen-phase pipeline on the current CUDA stream, alternating
between cuBLAS GEMMs (six `at::matmul_out` calls) and three custom
CUDA fused-elementwise kernels. Intermediates are allocated via
`torch::empty()` and dropped as soon as the dependency chain frees
them, keeping the peak resident memory below the 3090 Ti's 24 GiB
on the workloads where the reference itself fits.

| # | Operation | Output | Kernel |
|---|-----------|--------|--------|
| 1 | grad_gated_output = grad_output @ down_weight | [B, S, I] | at::matmul_out |
| 2 | gated_output_ws = activated_gate * up_output  | [B, S, I] | custom swiglu_mul |
| 3 | grad_down_weight = grad_output.T @ gated_output_ws | [H, I] | at::matmul_out |
| 4 | (drop gated_output_ws) | | |
| 5 | fused: grad_up_output, grad_gate_output from (grad_gated_output, up_output, activated_gate, gate_output) | [B, S, I] × 2 | custom swiglu_backward_elementwise |
| 6 | (drop grad_gated_output) | | |
| 7 | grad_x_up_ws = grad_up_output @ up_weight | [B, S, H] | at::matmul_out |
| 8 | grad_up_weight = grad_up_output.T @ x | [I, H] | at::matmul_out |
| 9 | (drop grad_up_output) | | |
| 10 | grad_x_gate_ws = grad_gate_output @ gate_weight | [B, S, H] | at::matmul_out |
| 11 | grad_gate_weight = grad_gate_output.T @ x | [I, H] | at::matmul_out |
| 12 | (drop grad_gate_output) | | |
| 13 | grad_x = grad_x_gate_ws + grad_x_up_ws | [B, S, H] | custom elementwise_add |

The fused phase-5 kernel computes `silu'(gate_output)` in FP32 as
`sigmoid(g) * (1 + g * (1 - sigmoid(g)))`, then folds the
`grad_activated_gate = grad_gated_output * up_output` and
`grad_up_output = grad_gated_output * activated_gate` multiplications
into the same bf16x2 grid-stride pass — one global-memory traversal
of the four input streams replaces four separate elementwise kernels.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-
checked and natively-correctness-validated on. Final SOL scoring is
on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060): the six
  GEMMs are heavyweight on the I=53248 side; B200's HBM3e bandwidth
  reshapes the bandwidth-vs-compute envelope substantially relative
  to the 3090 Ti / 5060 GDDR-class memory subsystems.
- DSMEM and cluster-cooperative GEMMs (sm_100a-only) are out of
  scope at the cuBLAS-dispatch portability level; cuBLAS picks an
  appropriate kernel on each host transparently.
- L2 carveout / partition: the largest workloads at B*S=8192 have
  per-intermediate footprints near 900 MiB, well outside the 3090
  Ti's 6 MB L2 — the GEMMs stream through HBM3e / GDDR6X.
- TCGen05 / TMEM divergence: cuBLAS on B200 picks the TCGen05 /
  TMEM tensor-core path; on the 3090 Ti and 5060 the same call
  resolves to Ampere / Ada tensor cores. Portability is at the
  call-site level, not the schedule level.
- Memory ceiling: with peak resident memory around 15 GiB at the
  largest workloads, the reference itself OOMs on the 3090 Ti for
  the heaviest workload entries. INVALID_REFERENCE on those rows
  is a host-resource gap, not a kernel correctness issue, and would
  not manifest on the 192 GiB B200 host.
