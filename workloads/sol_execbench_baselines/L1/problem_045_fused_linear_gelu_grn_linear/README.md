# problem_045_fused_linear_gelu_grn_linear

| Field                       | Value                                                                                            |
|-----------------------------|--------------------------------------------------------------------------------------------------|
| Problem                     | Fused MLP block with GELU + Global Response Normalization (ConvNextV2 pattern, NHWC)             |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_045_fused_linear_gelu_grn_linear.md`        |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/045_fused_linear_gelu_grn_linear/`                              |
| Variant                     | `cuda_cpp`                                                                                       |
| Evidence                    | `Profiled-local-sm_120`                                                                                |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                                 |
| Local speedup vs reference  | 0.97 - 1.01x on sm_86 across 14/14 PASSED workloads (exact match: max_rel_err = 0, max_abs_err = 0; diagnostic) |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 - 0.5+>                                 |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                                |

## Approach

Pure ATen / cuBLAS dispatch — NO custom CUDA kernels. The forward
computes one output through two `at::linear` calls (each
dispatching cublasSgemm 2D fp32 OP_N OP_T) plus a GELU activation
and a Global Response Normalization block:

1. Expansion Linear (with bias):
   - `x = at::linear(hidden_states, pwconv1_weight, pwconv1_bias)`
     (NHWC [B, H, W, 128] @ [512, 128].T -> [B, H, W, 512]; F.linear
     flattens the leading dims for the matmul and reshapes back).
2. GELU activation (erf-based, the PyTorch default):
   - `x = at::gelu(x)`.
3. Global Response Normalization:
   - `global_features = at::linalg_vector_norm(x, ord=2,
     dim=std::vector<int64_t>{1, 2}, keepdim=true)`
     (L2 norm across the spatial H and W axes; result is
     [B, 1, 1, 512]). The `dim` is a typed vector so the
     int-axis overload binds unambiguously (a bare brace-init
     would be ambiguous against the named-axis overload).
   - `channel_mean = global_features.mean(dim=-1, keepdim=true)`
     (channel-wise mean; result is [B, 1, 1, 1]).
   - `norm_features = global_features / (channel_mean + eps)`.
   - `x = grn_weight * (x * norm_features) + grn_bias + x`
     (affine multiply with grn_weight on the scaled values, add
     grn_bias, add the pre-GRN `x` as a residual).
4. Projection Linear (with bias):
   - `output = at::linear(x, pwconv2_weight, pwconv2_bias)`
     (NHWC [B, H, W, 512] @ [128, 512].T -> [B, H, W, 128]; same
     cublasSgemm 2D fp32 OP_N OP_T family at a different shape).

Const axes pinned as `constexpr`: `dim = 128`, `hidden_dim = 512`
(the ConvNextV2 4x expansion factor). Every indexed tensor is
shape-checked against these values. CUDA guard from the first
input. Launch on `at::cuda::getCurrentCUDAStream()` per CLAUDE.md
Rule 7.

Dispatch family note: both `at::linear` calls bind the same
`cublasSgemm_2D_fp32_OP_N_OP_T` family that batch-041
(problem_074) introduced into
`.humanize/corpus-expansion/verified_dispatch_families.toml` with
`verification_status = "pending_verification"`. batch-042 is the
second witness for the same family at different operand shapes
(NHWC 4D input flattened to [B*H*W, dim] for the matmul, vs
batch-041's 3D [B, S, hidden_size] input). The family remains
pending until cuBLAS-logger evidence promotes it to `verified`.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) compiles and is expected to smoke-run this baseline;
final SOL scoring is on B200 (sm_100a). Differences worth noting for this
pattern:

- HBM3e vs GDDR6X bandwidth: the fused MLP-GRN block is memory-bound on the
  two weight reads (pwconv1_weight + pwconv2_weight) and the GRN reductions;
  the 3090 Ti's bandwidth ceiling makes local wall-time a weak proxy for the
  B200 SOL Score. Local timing primarily validates correctness and
  launch-config sanity.
- DSMEM / cluster-cooperative reduction (sm_100a-only) are not used here;
  the two cublasSgemm dispatches go through cuBLAS's standard kernel
  selection on every arch, and the GRN reductions run as standard ATen
  reduction kernels.
- L2 carveout: B200's larger L2 absorbs both weight tensors
  (pwconv1: 512x128 = 256 KiB, pwconv2: 128x512 = 256 KiB) easily; the
  3090 Ti's smaller L2 also holds them comfortably for this shape range.
  GDDR7 does not apply here (the 3090 Ti uses GDDR6X; sm_120 would be the
  GDDR7 reference if a future cloud host is allocated).
- TCGen05 / TMEM: this template does not use 5th-gen tensor cores or the
  dedicated TMEM register surface; the fp32 sgemm path is the conservative
  PyTorch dispatch and is not exposed to sm_100a-only TCGen05 / TMEM codegen
  differences. A future tensor-core-aware ConvNextV2 variant (which would
  touch TCGen05 / TMEM on B200) is out of scope for the current template.
