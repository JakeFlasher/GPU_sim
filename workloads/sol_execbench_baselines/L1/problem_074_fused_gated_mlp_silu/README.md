# problem_074_fused_gated_mlp_silu

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | Fused gated MLP with SiLU activation (gate-up Linear -> chunk -> SiLU-gate -> down Linear) |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_074_fused_gated_mlp_silu.md`           |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/074_fused_gated_mlp_silu/`                                 |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | 0.66 - 1.53x on sm_86 across 16/16 PASSED workloads (exact match: max_rel_err = 0, max_abs_err = 0; diagnostic) |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 - 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Pure ATen / cuBLAS dispatch — NO custom CUDA kernels. The forward
computes one output through two `at::linear` calls (each binding
to a fp32 2D `cublasSgemm` dispatch at the novel
`cublasSgemm_2D_fp32_OP_N_OP_T` family) plus an inline SiLU-gate
fusion:

1. Fused gate-up projection:
   - `up_states = at::linear(hidden_states, gate_up_weight)`
     ([B, S, hidden_size=1024] @ [intermediate_size_x2=8192,
     hidden_size=1024].T -> [B, S, intermediate_size_x2=8192]).
   - The implicit `.T` on the weight makes the cuBLAS dispatch
     transA=N, transB=T (the novel family).
2. Split + SiLU-gate fusion:
   - `(gate, up_split) = up_states.chunk(2, dim=-1)`
     (each: [B, S, intermediate_size=4096]).
   - `silu_gate = gate * at::sigmoid(gate)` (SiLU activation).
   - `gated = up_split * silu_gate` (gated multiply).
3. Down projection:
   - `output = at::linear(gated, down_weight)`
     ([B, S, intermediate_size=4096] @ [hidden_size=1024,
     intermediate_size=4096].T -> [B, S, hidden_size=1024]).
   - Same novel-family dispatch at a different shape.

Const axes pinned as `constexpr`: `hidden_size = 1024`,
`intermediate_size = 4096`, `intermediate_size_x2 = 8192`. Every
indexed tensor is shape-checked against these values. CUDA guard
from the first input. Launch on
`at::cuda::getCurrentCUDAStream()` per CLAUDE.md Rule 7.

Dispatch family note: the deployed baseline introduces
`cublasSgemm_2D_fp32_OP_N_OP_T` as a novel family entry under
`.humanize/corpus-expansion/verified_dispatch_families.toml`
with `verification_status = "pending_verification"`. The
inferred dispatch shape (F.linear forward on 2D fp32 contiguous
operands -> cublasSgemm transA=N transB=T) is documented in the
manifest's verification_evidence field. cuBLAS-logger promotion
to `verified` is a follow-on operator step.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) compiles and is expected to smoke-run this baseline;
final SOL scoring is on B200 (sm_100a). Differences worth noting for this
pattern:

- HBM3e vs GDDR6X bandwidth: the gated MLP is memory-bound on the two
  weight reads (gate_up_weight + down_weight); the 3090 Ti's bandwidth
  ceiling makes local wall-time a weak proxy for the B200 SOL Score.
  Local timing primarily validates correctness and launch-config sanity.
- DSMEM / cluster-cooperative reduction (sm_100a-only) are not used here;
  the two cublasSgemm dispatches go through cuBLAS's standard kernel
  selection on every arch.
- L2 carveout: B200's larger L2 absorbs the gate_up_weight tensor
  (8192 x 1024 fp32 = 32 MiB) plus the up_states intermediate; the
  3090 Ti's smaller L2 may evict between the two Linear dispatches.
  GDDR7 does not apply here (the 3090 Ti uses GDDR6X; sm_120 would be
  the GDDR7 reference if a future cloud host is allocated).
- TCGen05 / TMEM: this template does not use 5th-gen tensor cores or
  the dedicated TMEM register surface; the fp32 sgemm path is the
  conservative PyTorch dispatch and is not exposed to sm_100a-only
  TCGen05 / TMEM codegen differences. A future tensor-core-aware
  variant (which would touch TCGen05 / TMEM on B200) is out of scope.
