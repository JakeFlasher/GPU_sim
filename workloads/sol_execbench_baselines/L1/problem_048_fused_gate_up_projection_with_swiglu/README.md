# problem_048_fused_gate_up_projection_with_swiglu

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | bf16 Gemma3 fused gate + up projection followed by GELU-tanh activation and element-wise multiplication. `hidden_size = 3072` and `intermediate_size = 24576` pinned. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_048_fused_gate_up_projection_with_swiglu.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/048_fused_gate_up_projection_with_swiglu/`                 |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Three-launch chain in CUDA C++:

1. **bf16 cuBLAS GEMM** via `at::matmul(x, gate_proj.t())` →
   `gate_output[B, S, intermediate_size]`. cuBLAS dispatches a
   bf16 GEMM kernel using Sm80 bf16 tensor cores with fp32
   accumulator. The reference also uses `torch.matmul`, so the
   cuBLAS path matches the reference's K-reduction tile order
   exactly.

2. **bf16 cuBLAS GEMM** via `at::matmul(x, up_proj.t())` →
   `up_output[B, S, intermediate_size]`. Same template, separate
   workspace.

3. **Fused per-element GELU-tanh + bf16 multiply kernel**.
   For each `(b, s, j) ∈ [B, S, intermediate_size]`:

       g_fp32   = (float) gate_output[b, s, j]
       g3       = g_fp32 * g_fp32 * g_fp32
       arg      = sqrt(2/pi) * (g_fp32 + 0.044715 * g3)
       act_fp32 = 0.5 * g_fp32 * (1 + tanhf(arg))
       act_bf16 = __float2bfloat16(act_fp32)         # downcast
       up_bf16  = up_output[b, s, j]
       output[b, s, j] = __hmul(act_bf16, up_bf16)   # bf16 product

   The fp32 GELU + explicit bf16 downcast before the final
   multiply preserves the reference's exact dtype contract
   (`activated_gate.to(bf16) * up_output` ⇒ bf16 × bf16).
   Doing the multiply in fp32 and casting at the end produces a
   value subtly different from the bf16-multiply path the
   reference uses; the explicit downcast keeps the baseline
   within the workload tolerance band.

### Allocations

`gate_output` and `up_output` are allocated by `at::matmul`
implicitly (returning new bf16 tensors). The single output
`output[B, S, intermediate_size]` is pre-allocated by the
eval driver (DPS).

### Runtime-check audit

Every const axis from `definition.json` enforced on every input
and output tensor that the kernel indexes with that fixed
assumption:

| Tensor      | Runtime check                                              |
|-------------|------------------------------------------------------------|
| `x`         | `is_cuda`, `kBFloat16`, `is_contiguous`, `dim == 3`, `size(2) == hidden_size = 3072` |
| `gate_proj` | same dtype/contig, `dim == 2`, `size(0) == intermediate_size = 24576`, `size(1) == hidden_size = 3072` |
| `up_proj`   | same dtype/contig, `dim == 2`, `size(0) == intermediate_size = 24576`, `size(1) == hidden_size = 3072` |
| `output`    | same dtype/contig, `dim == 3`, `size(0) == x.size(0)`, `size(1) == x.size(1)`, `size(2) == intermediate_size = 24576` |

Destination-passing-style entry: `output` is pre-allocated by
the eval driver. Each output element is written exactly once
by the fused per-element kernel.

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was
compile-checked and natively-correctness-validated on. Final
SOL scoring is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5060):
  the larger weight tensors (`gate_proj` and `up_proj` each at
  `24576 × 3072 × 2 bytes = 144 MiB` bf16) substantially
  exceed the 3090 Ti's 6 MiB L2 partition. B200's HBM3e + larger
  L2 reshape the cross-CTA reuse pattern; cuBLAS may also
  select a different GEMM kernel on sm_100a's bf16 tensor
  cores.
- DSMEM and cluster-cooperative tile staging (sm_100a-only)
  are out of scope at this Sm80-portability level.
- TCGen05 / TMEM divergence: the two bf16 GEMMs run on Sm80
  bf16 tensor cores (mma.m16n8k16 bf16), not on B200's TCGen05
  / TMEM path. The fused per-element kernel is bandwidth-bound
  on three bf16 streams (gate_output read, up_output read,
  output write).
- L2 carveout / partition: workload-dependent. Per-launch
  weight read traffic is `2 × 24576 × 3072 × 2 = 288 MiB`
  bf16 — well over the 3090 Ti's 6 MiB L2 even at the smallest
  `B*S = 128` workload, so the L2 captures little cross-launch
  reuse on the local host.
- The bf16 K-reduction (K = 3072) is done by cuBLAS with an
  fp32 accumulator (`CUBLAS_COMPUTE_32F`), matching the
  reference's `torch.matmul` semantics. Using cuBLAS in both
  the deployed baseline AND the cohort-B probe ensures NCU
  measures the same kernel cuBLAS picks at runtime.
