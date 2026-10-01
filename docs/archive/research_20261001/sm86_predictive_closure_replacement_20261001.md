# SM86 predictive-closure replacement research

**Date:** 2026-10-01
**Scope:** RTX 3090 Ti / GA102 / SM86; native trace replay and NCU comparison.

## Decision

The retired mechanism-recovery objective is outside normal academic scope. The valid replacement is an observation-equivalent predictive model: retain hidden
mechanisms as latent candidate sets, model only resource/service effects that
can affect the declared output, and compare a separately named NCU endpoint
projection against held-out NCU labels.

This is a scope decision, not a completion claim: predict a declared measurement contract with error and uncertainty evidence while leaving unobservable fields black-box.

## Research basis

- [Accel-Sim](https://engineering.purdue.edu/tgrogers/publication/khairy-isca-2020/)
  demonstrates native SASS trace replay, configurable performance modeling, and
  counter-wise validation without a public full NVIDIA compiler.
- [Accel-Sim 2.0](https://github.com/purdue-aalp/accel-sim-framework-public)
  adds aggregate CUPTI PM time-series correlation through GPUVision. Its
  advertised H100 result is methodology context, not SM86 ground truth.
- [Huerta et al.](https://arxiv.org/abs/2503.20481) reverse-engineer modern
  issue/dependency behavior and validate aggregate execution cycles, but do not
  expose physical per-instruction routes.
- [Black-box GPU performance modeling](https://doi.org/10.1177/1094342020921340)
  calibrates interpretable models with parameterized measurement kernels rather
  than requiring proprietary microarchitectural knowledge.
- [PPT-GPU](https://github.com/NMSU-PEARL/PPT-GPU) and
  [Profiner](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14671)
  show that address traces, reuse profiles, and cache-flow models can be
  useful without cache-owner or DRAM-completion telemetry.
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
  documents that pipeline counts can be estimated rather than dynamically
  assigned, and that NCU duration uses profiler-specific WaitForIdle,
  semaphore, replay, serialization, and cache-control semantics.
- NVIDIA's public [physical-SM/SMSP issue](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1351)
  confirms the absence of a supported globally identifiable physical
  SM/SMSP time-series interface.
- The [PTX memory model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
  treats coherence order as runtime-determined and cache-policy modifiers as
  performance hints, supporting candidate write/read modeling rather than
  physical ownership claims.
- [CuAssembler](https://github.com/gpuocelot/cuasm) and
  [nvsass-disassembler](https://github.com/reoLantern/nvsass-disassembler)
  remain finite assembler/decoder tools; compiler, scheduler, linker, and
  CUDA-ABI internals stay outside the black-box modeling boundary.
- [FlashGPU-sim](https://arxiv.org/abs/2609.15311),
  [GPU-Tile-Sim](https://arxiv.org/abs/2607.11262), and
  [NeuSight](https://arxiv.org/abs/2407.13853) reinforce a useful modern
  pattern: dependency/tile/resource abstractions can predict performance, but
  their newer-GPU or AI-specific validation does not transfer automatically to
  GA102.

## Proposed model

For each dynamic event `e`, retain:

```text
route(e)    = candidate abstract resource pools
compiler(e) = trace/semantic realization domain
cache(e)    = candidate sector state and read-from writes
timing(e)   = simulator and observer endpoint domains
```

The route layer computes earliest/latest/expected issue under service-pool
capacity, initiation interval, register-bank demand, and queue contention. It
never emits a physical route label.

The cache layer replays sectors and reuse distance, tracks write epochs and
memory-scope ordering, and returns a candidate read-from set. Model-local
writeback and forwarding are conservation checks, not hardware acknowledgements.

The endpoint layer leaves simulator issue/event/tail frontiers separate from an
NCU observer projection. Cache-control, launch serialization, clock policy, and
profiler start/end semantics are explicit inputs.

## Calibration and scoring

Probe kernels calibrate positive service rates and cache-flow parameters. No
score NCU label is consumed as a feature. The score cohort is family-held-out
and labels are joined only after prediction. Report:

- primary cycles MAPE and absolute log error;
- endpoint-specific duration error;
- counter-wise instruction/cache traffic error;
- interval coverage and width;
- rank correlation and OOD refusal/interval widening.

The current local 15/15 cold and warm campaigns remain the detailed NCU
  comparison set. Existing self-authored projection results are implementation
  evidence only; they are not independent silicon validation or a general
  accuracy result.
