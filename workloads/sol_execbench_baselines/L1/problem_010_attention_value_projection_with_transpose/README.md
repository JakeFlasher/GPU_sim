# problem_010_attention_value_projection_with_transpose

| Field                       | Value                                                                                            |
|-----------------------------|--------------------------------------------------------------------------------------------------|
| Problem                     | GQA-style attention value projection with reshape + transpose to attention layout                |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_010_attention_value_projection_with_transpose.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/010_attention_value_projection_with_transpose/`                 |
| Variant                     | `cuda_cpp`                                                                                       |
| Evidence                    | `Profiled-local-sm_120`                                                                                |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                                 |
| Local speedup vs reference  | 0.93 - 1.01x on sm_86 across 16/16 PASSED workloads (exact match: max_rel_err = 0, max_abs_err = 0; diagnostic) |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 - 0.5+>                                 |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                                |

## Approach

Pure ATen / cuBLAS dispatch — NO custom CUDA kernels. The forward
computes one output through one `at::linear` call (cublasGemmEx
2D bf16 OP_N OP_T) plus a view + transpose + contiguous to
materialise the GQA-attention output layout:

1. Linear value projection:
   - `value_proj = at::linear(hidden_states, v_proj_weight)`
     ([B, S, 5120] @ [1024, 5120].T -> [B, S, 1024]). The
     implicit `.T` on the weight makes the cuBLAS dispatch
     transA = N, transB = T at the novel
     `cublasGemmEx_2D_bf16_OP_N_OP_T` family.
2. Reshape to attention layout:
   - `value_reshaped = value_proj.view(B, S, num_kv_heads=8, head_dim=128)`.
   - `value_states.copy_(value_reshaped.transpose(1, 2).contiguous())`
     materialises [B, num_kv_heads, S, head_dim] in DPS.

Const axes pinned as `constexpr`: `hidden_size = 5120`,
`num_kv_heads = 8`, `head_dim = 128`, `kv_out_dim = 1024`. Every
indexed tensor is shape-checked against these values. CUDA guard
from the first input. Launch on
`at::cuda::getCurrentCUDAStream()` per CLAUDE.md Rule 7.

Dispatch family note: the deployed baseline introduces
`cublasGemmEx_2D_bf16_OP_N_OP_T` as a novel family entry under
`.humanize/corpus-expansion/verified_dispatch_families.toml`
with `verification_status = "pending_verification"`. cuBLAS-logger
promotion to `verified` is a follow-on operator step.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) compiles and is expected to smoke-run this baseline;
final SOL scoring is on B200 (sm_100a). Differences worth noting for this
pattern:

- HBM3e vs GDDR6X bandwidth: value projection is memory-bound on the weight
  read and the contiguous-copy of the transposed view; the 3090 Ti's
  bandwidth ceiling makes local wall-time a weak proxy for the B200 SOL
  Score. Local timing primarily validates correctness and launch-config
  sanity.
- DSMEM / cluster-cooperative reduction (sm_100a-only) are not used here;
  the cublasGemmEx dispatch goes through cuBLAS's standard kernel selection
  on every arch.
- L2 carveout: B200's larger L2 absorbs the v_proj_weight tensor
  (1024 x 5120 bf16 = 10 MiB) more easily; the 3090 Ti's smaller L2 may
  partially evict. GDDR7 does not apply here (the 3090 Ti uses GDDR6X;
  sm_120 would be the GDDR7 reference if a future cloud host is allocated).
- TCGen05 / TMEM: this template does not use 5th-gen tensor cores or the
  dedicated TMEM register surface; the bf16 cublasGemmEx path on Ampere /
  Hopper uses the standard tensor cores, not the sm_100a-only TCGen05
  surface. A future tensor-core-aware variant (which would touch TCGen05 /
  TMEM on B200) is out of scope.
