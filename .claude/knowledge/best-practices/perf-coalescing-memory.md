---
Use when: deciding whether to refactor a CUDA kernel for global-memory coalescing, shared-memory bank-conflict avoidance, L2 locality, or cache-behavior prediction under the non-root measurement contract.
Primary raw source: references/NVIDIA_Manuals/best_practices/10_memory-optimizations.md.
Supporting sources: references/NVIDIA_Manuals/cuda_guide/02-02-04-01_coalesced-global-memory-access.md; references/NVIDIA_Manuals/best_practices/09_performance-metrics.md.
Repo touchpoints: workloads/simbench_bw/, tools/nvbit_tracer/, .claude/knowledge/measurement/nonroot-cloud-contract.md.
Out of scope: NCU workflows, hardware-counter hit-rate collection, occupancy launch tuning, and instruction-throughput tuning.
Version family: CUDA 13.x, compute capability 6.0+; re-check raw NVIDIA docs on CUDA upgrades.
---

# Memory Optimization and Cache Behavior

This card gives memory-system tuning facts that can be used without NCU or
hardware counters. In this fork, cache hit ratio is a prediction target, not an
NCU metric to read directly.

## Global-Memory Coalescing

For compute capability 6.0+:

- The global-memory access unit is 32 bytes.
- A warp touching four aligned 32-byte sectors for 4-byte sequential reads is
  optimal.
- A misaligned sequential warp often touches five sectors.
- Strided access can approach one sector per thread in the worst case.
- `cudaMalloc` allocations are at least 256-byte aligned.
- Use block sizes that are multiples of 32 to preserve warp alignment.

Static SASS/PTX, source indexing patterns, and NVBit dynamic addresses are
allowed ways to build coalescing features.

## L2 and Persistence

L2 is shared across SMs. Ampere+ persistence APIs remain useful, but in this
project they should be treated as ordinary CUDA runtime configuration, not as a
profiler-counter workflow. Record the chosen access-policy-window settings in
the workload manifest when they are part of an experiment.

## Shared Memory

For modern NVIDIA GPUs:

- 32 banks, each 32 bits wide.
- Successive 32-bit words map to successive banks.
- Uniform access to one word broadcasts.
- Multiple different words in the same bank serialize.

Common fixes:

- Pad tiled arrays, for example `tile[TILE][TILE + 1]`.
- Load coalesced global spans into shared memory, then read the required
  strided pattern from shared memory.
- Use explicit `__syncthreads()` or `__syncwarp()` based on producer/consumer
  scope.

## Local Memory and Spills

Local memory lives in device memory and is usually caused by register spills,
large stack arrays, or non-constant indexing into local arrays. Allowed checks:

- `nvcc --ptxas-options=-v`
- static SASS inspection for `LDL` / `STL`
- elapsed-time sensitivity to register-budget variants

Do not use NCU local-memory metrics in this fork.

## Cache-Hit Prediction Inputs

Admissible features include:

- Static load/store opcode mix.
- Address-space qualifiers and cache operators in PTX/SASS.
- Dynamic address traces from bounded NVBit instrumentation.
- Reuse-distance and stride features computed offline.
- CUDA-event or CUPTI Activity elapsed-time response under controlled workload
  variants.
- Read-only environmental covariates.

## See Also

- `.claude/knowledge/measurement/nonroot-cloud-contract.md`
- `.claude/knowledge/best-practices/perf-occupancy-launch.md`
- `.claude/knowledge/blackwell/blackwell-tuning.md`
- `.claude/knowledge/cuda/cuda-clusters-dsmem.md`
- `workloads/simbench_bw/README.md`

## Escalate to Raw When

- You need the full coalescing examples or V100 bandwidth figures from the raw
  best-practices guide.
- You need NUMA, UVM, zero-copy, pinned-memory, or multi-stream transfer details
  beyond this summary.
