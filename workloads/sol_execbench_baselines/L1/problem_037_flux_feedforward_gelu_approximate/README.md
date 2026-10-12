# problem_037_flux_feedforward_gelu_approximate

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | fp32 FLUX feedforward (Linear → GELU-tanh → Linear). `hidden_dim=3072`, `mlp_hidden_dim=12288` pinned. `batch_size` and `seq_len` vary. From the FLUX.1-dev transformer block (76 calls per forward pass). |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_037_flux_feedforward_gelu_approximate.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/037_flux_feedforward_gelu_approximate/`                    |
| Variant                     | `cutlass`                                                                                   |
| Evidence                    | `Profiled-local-sm_120`                                                                           |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                            |
| Local speedup vs reference  | diagnostic only                                                                             |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Four-kernel chain in fp32:

1. **CUTLASS GEMM 1** — `workspace[B*S, 12288] =
   hidden_states[B*S, 3072] @ fc1_weight.T`.
   Sm80 TensorOp tf32 mma (`mma.m16n8k8` tf32 instruction).
2. **GELU-tanh + bias kernel** (in-place on `workspace`) —
   per-(b, s, n) thread reads `workspace[bs, n]`, adds
   `fc1_bias[n]`, applies the GELU-tanh approximation:
   ```
   gelu_tanh(z) = 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))
   ```
   then writes back to `workspace[bs, n]`.
3. **CUTLASS GEMM 2** — `output[B*S, 3072] =
   workspace[B*S, 12288] @ fc2_weight.T`. Same template.
4. **Bias-add kernel** (in-place on `output`) — per-(b, s, n)
   thread reads `output[bs, n]`, adds `fc2_bias[n]`, writes
   back.

CUTLASS Sm80 TensorOp fp32 instantiation:

- `ElementInputA = ElementInputB = ElementOutput = float`,
  `ElementAccumulator = float`.
- `OpClassTensorOp`, `Sm80`.
- `ThreadblockShape<128, 128, 16>` (fp32 fits 16 elements per
  tile-K column).
- `WarpShape<64, 64, 16>`, `InstructionShape<16, 8, 8>` —
  `mma.m16n8k8` tf32 mma instruction on Sm80.
- `NumStages = 3`, default `LinearCombination<float, 4, float,
  float>` epilogue (`AlignmentC = 4` for fp32 = 128-bit vector
  stores; 12288 mod 4 = 0 and 3072 mod 4 = 0).

The fp32 TensorOp path uses tf32 (10-bit mantissa) for the
internal mma accumulation, matching cuBLAS sgemm's default
behavior on Ampere. This is what `F.linear` invokes for fp32
on CUDA-Ampere by default. The reference and the baseline
therefore share the same numerical path; tolerance budget
(`max_atol = 0.0022-0.0024`, `max_rtol = 1e-5`) absorbs the
remaining ULP drift from differing tile orders.

### Workspace

`workspace[M = B*S, mlp_hidden_dim = 12288]` is allocated by
the launcher via `torch::empty(...)` and reused across the
GEMM 1 output and the GELU-tanh+bias in-place stage. The
workspace is freed at scope exit; it represents the largest
intermediate (`M * 12288 * 4 B`).

### Runtime-check audit

Every const axis from `definition.json` is enforced on every
input and output tensor the kernel indexes with that fixed
assumption:

| Tensor          | Runtime check                                              |
|-----------------|------------------------------------------------------------|
| `hidden_states` | `is_cuda`, `kFloat32`, `is_contiguous`, `dim == 3`, `size(2) == hidden_dim = 3072` |
| `fc1_weight`    | same dtype/contig, `dim == 2`, `size(0) == mlp_hidden_dim = 12288`, `size(1) == hidden_dim = 3072` |
| `fc1_bias`      | same dtype/contig, `dim == 1`, `size(0) == mlp_hidden_dim = 12288` |
| `fc2_weight`    | same dtype/contig, `dim == 2`, `size(0) == hidden_dim = 3072`, `size(1) == mlp_hidden_dim = 12288` |
| `fc2_bias`      | same dtype/contig, `dim == 1`, `size(0) == hidden_dim = 3072` |
| `output`        | same dtype/contig, `dim == 3`, `size(0) == hidden_states.size(0)`, `size(1) == hidden_states.size(1)`, `size(2) == hidden_dim = 3072` |

Destination-passing-style entry: `output` is pre-allocated by
the eval driver. The launcher writes each output element
exactly once (the GEMM 2 writes the fp32 sum + the bias-add
kernel adds the per-channel `fc2_bias`).

## Architecture-coupling caveats

The local 3090 Ti (sm_86) is the host this baseline was
compile-checked and natively-correctness-validated on. Final
SOL scoring is on B200 (sm_100a). Differences worth noting:

- HBM3e (B200) vs GDDR6X (3090 Ti) vs GDDR7 (sm_120 5090):
  the two GEMMs and the GELU/bias kernels are bandwidth-bound
  on the weight streams (`fc1_weight` is 144 MiB,
  `fc2_weight` is 144 MiB fp32). B200's HBM3e roughly 8x
  bandwidth of the 3090 Ti's GDDR6X. Both 144 MiB weight
  tensors far exceed the 3090 Ti's 6 MiB L2 and the B200's
  L2 partition; pure HBM streaming for the B-stream.
- DSMEM and cluster-cooperative B-tile staging (sm_100a-only)
  are out of scope at this Sm80-portability level.
- TCGen05 / TMEM divergence: Sm80 mma.m16n8k8 tf32 is used;
  the B200 TCGen05 / TMEM fp32 path is NOT exercised here.
- L2 carveout / partition: workload-dependent. The intermediate
  `workspace[M, 12288]` fp32 is `M * 48 KiB`; for canonical
  small-M workloads (B=1, S=1024 → M=1024) it's 48 MiB —
  exceeds L2; for M ≤ 128 it fits.
- TF32 mma: Sm80 truncates fp32 inputs to 10-bit mantissa
  before each `mma.m16n8k8` step. Output stays fp32. cuBLAS
  sgemm does the same by default on Ampere; matches
  `F.linear` behavior. The workload tolerance budget absorbs
  the remaining ULP drift.
