# problem_063_attention_output_reshape_and_projection

| Field                       | Value                                                                                            |
|-----------------------------|--------------------------------------------------------------------------------------------------|
| Problem                     | Fused attention output reshape + projection (multi-head -> per-token -> hidden_size projection)  |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_063_attention_output_reshape_and_projection.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/063_attention_output_reshape_and_projection/`                   |
| Variant                     | `cuda_cpp`                                                                                       |
| Evidence                    | `Profiled-local-sm_120`                                                                                |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                                 |
| Local speedup vs reference  | 0.91 - 1.16x on sm_86 across 16/16 PASSED workloads (exact match: max_rel_err = 0, max_abs_err = 0; diagnostic) |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 - 0.5+>                                 |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                                |

## Approach

Pure ATen / cuBLAS dispatch — NO custom CUDA kernels. The forward
computes one output through one `at::matmul` call (cublasGemmEx
2D bf16 OP_N OP_T with explicit weight transpose) plus a
transpose + contiguous + view chain that unwinds the multi-head
attention layout to the flat per-token layout the projection
consumes:

1. Multi-head -> per-token layout flip:
   - `attn_output_reshaped = attn_output.transpose(1, 2).contiguous().view({B, S, intermediate_size=16384})`
     (the `.contiguous()` triggers a full physical copy of the
     transposed view; the `.view()` then reshapes the contiguous
     buffer to the flat per-token shape).
2. Projection:
   - `output = at::matmul(attn_output_reshaped, o_proj_weight.t())`
     ([B, S, 16384] @ [7168, 16384].T -> [B, S, 7168]). The
     explicit `.t()` on the weight binds the cuBLAS dispatch to
     transA = N, transB = T (the
     `cublasGemmEx_2D_bf16_OP_N_OP_T` family pinned by batch-043
     + batch-044). Copied into the DPS output buffer.

Const axes pinned as `constexpr`: `num_heads = 128`,
`v_head_dim = 128`, `intermediate_size = 16384`,
`hidden_size = 7168`. Every indexed tensor is shape-checked
against these values. CUDA guard from the first input. Launch on
`at::cuda::getCurrentCUDAStream()` per CLAUDE.md Rule 7.

Dispatch family note: this baseline is the second witness for
the `cublasGemmEx_2D_bf16_OP_N_OP_T` family declared in
`.humanize/corpus-expansion/verified_dispatch_families.toml`
with `verification_status = "pending_verification"`. cuBLAS-logger
promotion to `verified` is a follow-on operator step (the two
landed witnesses at distinct operand shapes provide
replay-ready evidence).

## Architecture-coupling caveats

The local 3090 Ti (sm_86) compiles and is expected to smoke-run this baseline;
final SOL scoring is on B200 (sm_100a). Differences worth noting for this
pattern:

- HBM3e vs GDDR6X bandwidth: the output projection is memory-bound on
  the o_proj_weight read (7168 x 16384 bf16 = 224 MiB -- large weight
  matrix); the 3090 Ti's bandwidth ceiling makes local wall-time a weak
  proxy for the B200 SOL Score. The contiguous-copy of the transposed
  attention layout is also bandwidth-bound and dominates at small B*S.
- DSMEM / cluster-cooperative reduction (sm_100a-only) are not used here;
  the cublasGemmEx dispatch goes through cuBLAS's standard kernel
  selection on every arch.
- L2 carveout: B200's larger L2 can hold partial tiles of the large
  o_proj_weight matrix more effectively; the 3090 Ti's smaller L2 must
  re-stream from HBM. GDDR7 does not apply here (the 3090 Ti uses
  GDDR6X; sm_120 would be the GDDR7 reference if a future cloud host
  is allocated).
- TCGen05 / TMEM: this template does not use 5th-gen tensor cores or
  the dedicated TMEM register surface; the bf16 cublasGemmEx path on
  Ampere uses the standard tensor cores, not the sm_100a-only TCGen05
  surface. A future tensor-core-aware variant is out of scope.
