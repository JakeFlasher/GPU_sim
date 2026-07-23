# problem_067_flash_attention_gqa_ultralong

| Field                       | Value                                                                  |
|-----------------------------|------------------------------------------------------------------------|
| Problem                     | <one-line description from the markdown spec>                          |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_067_flash_attention_gqa_ultralong.md`                |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L1/067_flash_attention_gqa_ultralong/`                  |
| Variant                     | `cuda_cpp`                                                           |
| Evidence                    | `Compile-checked-sm_86`                                  |
| Local speedup vs reference  | <diagnostic only; `< 1.0` is acceptable>                               |
| SOL Score (online judge)    | <filled after Phase-2 submission; target band 0.3 – 0.5+>              |
| Counter-free artifacts      | <which of {nvbit, cupti, nsys, static, sanitizer} emitted ok>        |

## Approach

<hand-port the official sol-execbench Python reference (and any
matching official Python solution under sol-execbench/examples/) into
this *.cu template; then describe the tiling / schedule / fusion
strategy / precision path here. Template solutions are profiling
targets, not optimized kernels.>

## Architecture-coupling caveats

The local NCU-permitted host (this scaffold detected `sm_86`)
compiles and smoke-runs this baseline; Phase-2 counter-free profiling
and SOL Score submission run on B200 (sm_100a). Differences worth
noting for this pattern:

- HBM3e vs GDDR6X / GDDR7 bandwidth: bandwidth-bound kernels have a
  different ceiling on B200 than on the local consumer GPU; local
  wall-time is a weak proxy for B200 behavior.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are out of
  scope for local consumer-arch builds.
- L2 carveout: B200's larger L2 may absorb tensors the local consumer
  GPU's smaller L2 cannot.
- TCGen05 / TMEM divergence: 5th-generation tensor cores and the
  dedicated TMEM register surface only exist on sm_100a; document any
  divergence here when the kernel touches them.
