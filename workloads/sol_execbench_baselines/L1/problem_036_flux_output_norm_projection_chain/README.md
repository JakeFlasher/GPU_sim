# problem_036_flux_output_norm_projection_chain

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | fp32 FLUX final-output processing chain combining AdaLayerNormContinuous + SiLU(temb) → modulation projection → adaptive modulation → final small-N output projection. Captured from FLUX.1-Kontext-dev. `inner_dim = 3072`, `output_dim = 64`, `modulation_dim = 6144` pinned. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_036_flux_output_norm_projection_chain.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/036_flux_output_norm_projection_chain/`                    |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Four-launch chain in fp32 CUDA C++. Combines the
AdaLayerNormContinuous pattern landed in the cohort's prior
slugs with a SiLU pre-projection on the timestep embeddings and
a final small-N=64 output projection via cuBLAS.

The reference algorithm is:

    mean       = hidden_states.mean(dim=-1, keepdim=True)
    var        = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
    hs_norm    = (hidden_states - mean) / sqrt(var + eps)
    temb_silu  = temb * sigmoid(temb)
    modulation = F.linear(temb_silu, linear_weight, linear_bias)
                 # [B, modulation_dim = 6144]
    shift      = modulation[:, :inner_dim]
    scale      = modulation[:, inner_dim:]
    hs_mod     = hs_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    output     = F.linear(hs_mod, proj_out_weight, proj_out_bias)
                 # [B, S, output_dim = 64]

### Kernel 1 — per-element SiLU on temb

Grid-stride loop over `[B, inner_dim]`. Each thread reads a
single fp32, computes `sigmoid(x) = 1 / (1 + exp(-x))` via
`__expf`, and writes `x * sigmoid(x)` into the
`temb_silu[B, inner_dim]` workspace. The input `temb` is not
mutated.

### Kernel 2 — Small-M Linear: `temb_silu → modulation[B, 6144]`

Per-(b, n_tile) CTA over Grid
`(B, modulation_dim / kWarpsPerCtaLinear = 6144 / 8 = 768)`.
BLOCK_THREADS = 256 (8 warps of 32 lanes).

- Each CTA shared-memory-caches `temb_silu[b, :K = 3072]` once
  (12 KiB fp32) so the 8 warps reuse it across their 8
  K-reductions.
- One warp per output element with `__shfl_xor_sync`
  K-reduction over `inner_dim = 3072`; lane 0 writes
  `partial + linear_bias[n]`.

Same small-M Linear structural pattern as the cohort's prior
small-M Linear kernels (where the canonical `batch_size`
1..32 leaves CUTLASS standard ThreadblockShapes severely
underutilized).

### Kernel 3 — Fused LN + modulation per-(b, s) row

Per-(b, s) CTA over Grid `(B, S)`; BLOCK_THREADS = 256
stride-loops over `inner_dim = 3072`.

**Pass A**: each thread accumulates per-row partial sum and
partial sum-of-squares of `hidden_states[b, s, :]`.
Warp-shuffle reduces within each warp; 8-element shared-
memory tree reduces across warps. Thread 0 computes
`mean = sum / inner_dim`, `var = sum_sq / inner_dim -
mean^2`, `rstd = rsqrtf(var + eps)`, and broadcasts both
back through shared memory.

**Pass B**: each thread `d` reads `hidden_states[b, s, d]`,
`modulation[b, d]` (**shift** — first half per the
reference's `modulation[:, :inner_dim]` chunk),
`modulation[b, d + inner_dim]` (**scale** — second half);
writes `hs_mod[b, s, d] = ((x - mean) * rstd) * (1 + scale) +
shift` into a workspace.

Note: this slug puts shift **first** in the modulation chunk;
the cohort's prior AdaLayerNormContinuous slug
(`problem_080`) put scale first. The kernel must use the
correct chunk order to match the reference.

Both passes use per-row reductions only (within a single
CTA). No cross-CTA reduction tree means no accumulation-
order drift against PyTorch's reduction order.

### Kernel 4 — cuBLAS fp32 GEMM: `output = hs_mod @ proj_out_weight.t() + proj_out_bias`

Dispatched via `at::linear_out(output, hs_mod, proj_out_weight,
proj_out_bias)`. cuBLAS picks an sgemm kernel suited to the
small-N = 64 shape; the reference also uses `F.linear`, so the
cuBLAS path matches the reference's dispatch exactly.

### Allocations

`temb_silu[B, inner_dim]`, `modulation[B, modulation_dim]`,
and `hs_mod[B, S, inner_dim]` are launcher-allocated
`torch::empty` fp32 workspaces. The single output
`output[B, S, output_dim]` is pre-allocated by the eval
driver (DPS) and written via `at::linear_out`.

### Runtime-check audit

Every const axis from `definition.json` enforced on every
input and output tensor that the kernel indexes with that
fixed assumption:

| Tensor            | Runtime check                                              |
|-------------------|------------------------------------------------------------|
| `hidden_states`   | `is_cuda`, `kFloat32`, `is_contiguous`, `dim == 3`, `size(2) == inner_dim = 3072` |
| `temb`            | same dtype/contig, `dim == 2`, `size(0) == hidden_states.size(0)`, `size(1) == inner_dim = 3072` |
| `linear_weight`   | same dtype/contig, `dim == 2`, `size(0) == modulation_dim = 6144`, `size(1) == inner_dim = 3072` |
| `linear_bias`     | same dtype/contig, `dim == 1`, `size(0) == modulation_dim = 6144` |
| `proj_out_weight` | same dtype/contig, `dim == 2`, `size(0) == output_dim = 64`, `size(1) == inner_dim = 3072` |
| `proj_out_bias`   | same dtype/contig, `dim == 1`, `size(0) == output_dim = 64` |
| `eps`             | scalar fp64 → cast to fp32 at the launcher boundary |
| `output`          | same dtype/contig, `dim == 3`, `size(0) == B`, `size(1) == S`, `size(2) == output_dim = 64` |

Destination-passing-style entry: `output` is pre-allocated by
the eval driver. Each output element is written exactly once
by `at::linear_out`.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was
compile-checked and natively-correctness-validated on. Final
SOL scoring is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5090):
  The Kernel 2 weight matrix is `6144 × 3072 × 4 = 72 MiB`
  fp32 — well over the 3090 Ti's 6 MiB L2 partition. B200's
  larger L2 will capture more cross-CTA weight reuse;
  Kernel 3 is dominated by per-(b, s) row traffic with no
  cross-row dependency.
- DSMEM and cluster-cooperative tile staging (sm_100a-only)
  are out of scope at this Sm80-portability level.
- TCGen05 / TMEM divergence: Kernel 2's K=3072 reduction is
  done in fp32 with warp shuffles (no tensor cores); Kernel 3
  is pure elementwise with per-row reduction. The cuBLAS
  Kernel 4 dispatches an fp32 sgemm kernel that uses tf32
  mma on Ampere — on B200 it would use TCGen05 / TMEM.
- L2 carveout / partition: workload-dependent. For canonical
  workloads with B up to 32 and S up to 4096, total
  `hidden_states` working set is up to `32 × 4096 × 3072 ×
  4 = 1.5 GiB` fp32 — far over any GPU L2. The small Kernel 4
  output (`output_dim = 64`) means the final GEMM is
  bandwidth-light by comparison.
- The Kernel 2 weight stream is the heaviest read pattern.
  Total weight traffic per launch ≈ 72 MiB; weight read once
  per CTA, but the same B-row of weight rows is read for
  every batch b (cross-b reuse via the 3090 Ti's L2 is
  bounded by the L2 footprint).
