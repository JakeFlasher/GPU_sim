# problem_004_fused_residual_rms_mlp

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | Fused residual + RMSNorm + SwiGLU MLP block. hidden_size=16384, intermediate_size=53248 pinned. Captured from Hermes-4-405B. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_004_fused_residual_rms_mlp.md`         |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L2/004_fused_residual_rms_mlp/`                               |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Compile-checked-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                             |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

The pipeline is decomposed into five phases on the current CUDA
stream:

1. **fused_residual_rms_kernel** (custom CUDA). One CTA per (b, s)
   row, 256 threads strided across `hidden_size = 16384` in bf16x2
   pairs. Pass 1 accumulates `(residual + hidden_states)^2` per row
   via warp-shuffle + cross-warp shared reduction; pass 2 re-reads
   the row plus the bf16 `norm_weight` and writes the normalized
   `norm_out` back as bf16. The two-pass design avoids staging a
   full 16384-element fp32 copy of each row in shared memory (which
   would exceed the per-SM budget at this hidden size).
2. **gate projection** — `at::matmul_out(gate_out, norm_out, gate_proj_weight.t())`.
   cuBLAS handle owned by PyTorch; stream-bound to
   `at::cuda::getCurrentCUDAStream()` (CLAUDE.md Rule 5: cuBLAS is a
   host API).
3. **up projection** — `at::matmul_out(up_out, norm_out, up_proj_weight.t())`.
4. **swiglu_kernel** (custom CUDA). Elementwise grid-stride over
   `B * S * intermediate_size` bf16x2 pairs; computes
   `silu(gate_out) * up_out` and writes back into `gate_out`'s
   storage so no extra workspace is allocated.
5. **down projection** — `at::matmul_out(output, gate_out, down_proj_weight.t())`.
   Writes directly into the DPS output tensor.

The same pattern is used by the already-landed
`problem_030_flux_concatenated_sequence_processing_with_split`
baseline (`at::matmul_out` for the linear projections, custom
kernels for the elementwise fusions).

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was compile-
checked and natively-correctness-validated on. Final SOL scoring is
on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5090): the three
  projection GEMMs are bandwidth-bound for the I=53248 weight reads
  (~1.6 GiB per weight matrix). B200's HBM3e (~8 TB/s) reshapes
  achievable arithmetic intensity vs. the 3090 Ti's GDDR6X
  (~1 TB/s). The custom kernels are also bandwidth-bound on the
  norm_out / gate_out / up_out streams.
- DSMEM and cluster-cooperative GEMMs (sm_100a-only) are out of
  scope at the cuBLAS-dispatch portability level — the underlying
  cuBLAS heuristic on B200 will use cluster kernels under the hood,
  but the host invocation surface is unchanged.
- L2 carveout / partition: B200's much larger L2 partition holds
  more of the gate_out / up_out activations than the 3090 Ti's
  6 MB L2 — the SwiGLU kernel will land in L2 on B200 and in HBM3e
  / GDDR6X spill on the 3090 Ti.
- TCGen05 / TMEM divergence: the three GEMMs are dispatched through
  cuBLAS, which on B200 will use the TCGen05 / TMEM tensor-core
  path automatically; on the 3090 Ti and 5090 it falls back to the
  Ampere / Ada tensor cores. The portability is at the call-site
  level, not at the schedule level.
