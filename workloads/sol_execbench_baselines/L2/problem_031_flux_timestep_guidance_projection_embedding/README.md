# problem_031_flux_timestep_guidance_projection_embedding

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | FLUX's combined timestep + text projection embedding (sinusoidal timestep embedding → SiLU-MLP, plus pooled-text linear projection, summed into conditioning). |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_031_flux_timestep_guidance_projection_embedding.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L2/031_flux_timestep_guidance_projection_embedding/`          |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | 1.09 – 1.13× on sm_86 (diagnostic only; cuBLAS-dominated)                                   |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

The reference fires three small matrix multiplications (timestep MLP
linear1, timestep MLP linear2, text embedder linear) plus ~8 small
elementwise kernels (timestep * 1000, broadcast multiply with `freqs`,
`sin`, `cos`, `cat`, `sigmoid`, multiply for SiLU, final add). At the
workload's small batch sizes (B ∈ [1, 20]) launch overhead is a
significant fraction of total runtime, so the win comes from
collapsing the elementwise chain rather than out-running cuBLAS at
the matmuls themselves.

Implementation:

- One fused CUDA kernel `problem_031_flux_timestep_embed_kernel`
  generates the `[B, 768]` `timestep_embed` directly from `timestep`
  and `freqs`. Each batch row is one CTA with 384 threads; each
  thread computes `arg = (timestep[row] * 1000) * freqs[tid]`, then
  uses `__sincosf` to emit `cos(arg)` to slot `[row, tid]` and
  `sin(arg)` to slot `[row, tid + 384]`. This collapses the
  reference's `timestep*1000 → broadcast mul → sin → cos → cat` chain
  (~5 launches) into one.
- `at::linear` for `linear1(timestep_embed, w1, b1)` (cuBLAS).
- One fused CUDA kernel `problem_031_flux_silu_kernel` applies
  in-place `x <- x * sigmoid(x)` over the `[B, 3072]` activations.
  Stock element-wise float32 stride-loop kernel; replaces the
  reference's separate `sigmoid` + `mul` launches.
- `at::linear_out(conditioning, …)` for `text_embedder(pooled_projections, w_text, b_text)`
  written directly into the DPS output buffer (cuBLAS).
- `at::linear` for `linear2(silu(x), w2, b2)` (cuBLAS), staged into
  a temporary `[B, 3072]` tensor.
- `conditioning.add_(timestep_mlp)` for the final accumulation
  (single vectorised add launch).

Total launch count: ~6 (2 custom kernels + 3 cuBLAS launches +
1 final add), vs the reference's ~11 launches. Local sm_86
result: **16/16 PASSED**, `speedup_factor` 1.09 - 1.13×. The
margin is tight because the three cuBLAS matmuls dominate runtime
(>95%) and we use cuBLAS in both paths; the elementwise-overhead
saving accounts for the entire speedup. No tensor cores are used
in this implementation — the matmuls are FP32 IEEE per the
problem's float32 dtype contract; FP32 IEEE compute on the local
sm_86 host is not on the tensor-core pipe.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) compiles and smoke-runs this baseline;
final SOL scoring is on B200 (sm_100a). Differences worth noting
for this pattern:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060)
  bandwidth: the matmul dominance means this kernel is
  compute-bound on the FP32 SM pipe, not bandwidth-bound. B200's
  HBM3e advantage matters less here than it would for an RMSNorm
  kernel.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are out
  of scope for sm_86; this kernel uses per-CTA registers only (no
  cross-CTA synchronisation).
- L2 carveout: the weights for `[3072, 3072]` linear2 (~36 MB
  in FP32) and `[3072, 768]` linear1 + text_embedder (~9 MB each)
  fit in B200's larger L2 carveout; on the 3090 Ti's 6 MiB L2 they
  are far larger than the cache. B200 may see different cuBLAS
  kernel selection because of this.
- TCGen05 / TMEM divergence: 5th-gen tensor cores and TMEM register
  surface only exist on sm_100a. This kernel does not use tensor
  cores at all (FP32 matmuls go through the FFMA SM pipe), so the
  TCGen05 / TMEM split is not exercised here. If a future variant
  upgrades the matmuls to use BF16 tensor cores via dtype casting,
  the sm_86 → sm_100a tensor-core ISA divergence would become
  relevant.
