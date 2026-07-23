# problem_030_flux_concatenated_sequence_processing_with_split

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | FLUX dual-stream linear projection (concat encoder + image streams → project → split back).|
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_030_flux_concatenated_sequence_processing_with_split.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L2/030_flux_concatenated_sequence_processing_with_split/`     |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                    |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                     |
| Local speedup vs reference  | Prior archive recorded 0.664 – 1.158× across 16 workloads (diagnostic only)                 |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

Per-stream `at::matmul_out` strategy reconstruction (the original
multi-launch source was never committed to git; this file is a
faithful re-implementation of the described strategy after the
2026-05-13 re-anchor to a correctness-only landing contract).

The reference does `concat → matmul → split`. We do **two direct
`at::matmul_out` calls**, one per output stream, sharing the
transposed weight view:

```
processed_encoder = encoder_hidden_states @ process_weight.t()
processed_hidden  = hidden_states         @ process_weight.t()
```

Both calls go through PyTorch's cuBLAS dispatch on
`at::cuda::getCurrentCUDAStream()`. The reference's concat + slice
memcpy traffic is eliminated at the cost of one additional cuBLAS
launch. There is no custom `__global__` kernel — the only CUDA work
is the cuBLAS dispatch path. This is a deliberate template-solution
choice: float32 IEEE matmul on a 3072×3072 weight is what cuBLAS is
already tuned for, and a hand-written fused TF32 / BF16 GEMM that
beats cuBLAS on every workload is out of scope for the project's
counter-free profiling framework deliverable (`CLAUDE.md` §
"Acceptance contract for *.cu template solutions").

`process_weight.t()` is a non-contiguous view; cuBLAS handles the
transpose internally via the operation type. No materialisation.

Per CLAUDE.md Rule 5 (CUDA C++ / CUTLASS C++ submission surface,
cuBLAS handled as a host API): cuBLAS handles are owned and cached
by PyTorch's CUDA backend; the kernel.cu launcher does not
instantiate its own cuBLAS handle. Default-stream discipline (Rule 7)
is preserved.

## Architecture-coupling caveats

The local 5060 (sm_120) compiles and smoke-runs this baseline;
Phase-2 counter-free profiling and SOL Score submission run on B200
(sm_100a). Differences worth noting for this pattern:

- **HBM3e vs GDDR7 bandwidth**: the projection reads `[B, S, 3072]`
  + `[3072, 3072]` weight, writes `[B, S, 3072]`, twice. For small
  `S` this is HBM-bound on both architectures; for large `S` it
  becomes compute-bound and B200's compute ceiling dominates. Local
  wall-time is a weak proxy for B200 wall-time at small `S`.
- **DSMEM and cluster-cooperative reduction** (sm_100a-only) are out
  of scope for this kernel — we delegate to cuBLAS, which on B200
  may itself use cluster-cooperative GEMM kernels under the hood.
  Re-validate on B200.
- **L2 carveout**: the 36-MiB `process_weight` matrix may fit in
  B200's L2 carveout but exceeds the 5060's L2; cuBLAS kernel
  selection differs between the two.
- **TCGen05 / TMEM divergence**: with float32 IEEE compute, this
  kernel does NOT use 5th-gen tensor cores on either arch (cuBLAS
  routes float32 IEEE through the FFMA SM pipe). If a future round
  upgrades to TF32 or BF16 with FP32 accumulators, the sm_120 →
  sm_100a tensor-core ISA divergence becomes load-bearing.
