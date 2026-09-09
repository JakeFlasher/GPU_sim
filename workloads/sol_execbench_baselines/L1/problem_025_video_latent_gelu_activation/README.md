# problem_025_video_latent_gelu_activation

| Field                       | Value                                                                                              |
|-----------------------------|----------------------------------------------------------------------------------------------------|
| Problem                     | FP32 GELU (tanh approximation) element-wise on video latent tensors; intermediate_size=4096       |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_025_video_latent_gelu_activation.md`         |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/025_video_latent_gelu_activation/`                                |
| Variant                     | `cuda_cpp`                                                                                         |
| Evidence                    | `Profiled-local-sm_120`                                                                            |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                             |
| Local speedup vs reference  | <pending local 3090 Ti run; diagnostic only>                                                       |
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                                   |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                                  |

## Approach

Pure element-wise GELU with the tanh approximation
`0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`. The
kernel streams `x` once and writes `output` once with float4
vectorized loads / stores (16-byte aligned by ATen). Each thread
processes one `float4` (four FP32 elements); a scalar tail kernel
handles the remainder when `numel() % 4 != 0`.

`intermediate_size = 4096` is fixed by `definition.json`;
`batch_size` and `seq_len` are variable. Launch is on
`at::cuda::getCurrentCUDAStream()` per CLAUDE.md Rule 7. Inputs are
validated for `is_cuda` / dtype / contiguity at the PyBind11
boundary; the output is written in destination-passing style.

The math uses the same constants the reference declares
(`sqrt(2/pi) = 0.7978845608028654`, `coeff = 0.044715`) and
`tanhf` for the transcendental — well within the workload's
`max_atol=1e-5`, `max_rtol=0.05` tolerance.

## Architecture-coupling caveats

The local 3090 Ti (sm_86 / GA102) compiles and runs this baseline;
final SOL scoring is on B200 (sm_100a / GB100). Differences worth
noting for this pattern:

- HBM3e vs GDDR6X bandwidth: this kernel is purely bandwidth-bound
  (one FP32 read + one FP32 write per element; the per-element
  arithmetic is ~8 ops including the SFU tanh, well below the
  compute ceiling). The 3090 Ti's GDDR6X bandwidth makes local
  latency a weak proxy for the B200 SOL Score.
- DSMEM and cluster-cooperative reduction (sm_100a-only): not used
  here. The pattern is independent element-wise with no shared
  reduction; B200 cluster support adds no behaviour for this
  kernel.
- L2 cache: working set per launch is `2 * numel() * 4` bytes; B200's
  larger L2 partitioning is not load-bearing here because the data
  is streamed once with no reuse.
- TCGen05 / TMEM divergence: this kernel does not touch tensor
  cores (no MACs). TCGen05 / TMEM are not exercised; the kernel is
  portable across sm_86 / 89 / 90 / 100 / 120 without
  architecture-dependent codegen.
- GDDR7 vs GDDR6X: the consumer 5060 has GDDR7. Bandwidth-bound
  kernels see a wall-clock difference, not a correctness one.

## Counter-free artifacts

Counter-free analytics bundle path:
`tools/sol_baselines_harness/.harness_traces/problem_025_video_latent_gelu_activation__cuda_cpp/`
(produced by the M4 surfaces orchestrator fanout; status pending
the local 3090 Ti pipeline run that is part of M4).
