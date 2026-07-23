---
Use when: choosing CUDA block size, grid size, register budget, shared-memory budget, or cluster launch shape under the non-root measurement contract.
Primary raw source: references/NVIDIA_Manuals/best_practices/11_execution-configuration-optimizations.md.
Supporting sources: references/NVIDIA_Manuals/cuda_guide/02-02-07_kernel-launch-and-occupancy.md; references/NVIDIA_Manuals/best_practices/10_memory-optimizations.md.
Repo touchpoints: workloads/simbench_bw/, .claude/knowledge/blackwell/blackwell-tuning.md, .claude/knowledge/measurement/nonroot-cloud-contract.md.
Out of scope: NCU Occupancy workflows, `nvidia-smi` compute-mode mutation, hardware-counter stall analysis, and simulator validation.
Version family: CUDA 13.x, compute capability 6.0+.
---

# Occupancy and Launch Configuration

Occupancy is active warps per SM divided by the architectural maximum. It is a
resource-capacity concept, not a goal by itself. Under this fork's contract,
occupancy decisions are evaluated through allowed timing, static build output,
runtime occupancy APIs, and workload variants, not NCU sections.

## What Gates Occupancy

| Gate | Source |
|---|---|
| Max warps per SM | Compute capability. |
| Max blocks per SM | Compute capability. |
| Registers per SM | Compute capability and per-thread register count. |
| Shared memory per SM/block | Compute capability plus static/dynamic shared memory. |
| Threads per block | Launch configuration. |
| Cluster size | Cluster-aware launch configuration and device ceiling. |

Use CUDA occupancy APIs for exact launch-specific answers:

- `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
- `cudaOccupancyMaxPotentialBlockSize`
- `cudaOccupancyAvailableDynamicSMemPerBlock`
- `cudaOccupancyMaxActiveClusters`

## Practical Rules

- Start with block sizes that are multiples of 32; 128 to 256 threads is often
  a useful first sweep range.
- Check `nvcc --ptxas-options=-v` for registers and spills.
- Use `__launch_bounds__` when a kernel has a fixed maximum block size.
- Use dynamic shared memory as a non-invasive occupancy throttle when measuring
  sensitivity.
- For SM120, remember the 48-warp SM ceiling and smaller shared-memory pool
  relative to B200.

## Non-Root Measurement Workflow

1. Compile with line info and `ptxas -v` where useful.
2. Record block, grid, register, shared-memory, stream, and cluster dimensions
   in the manifest.
3. Sweep launch parameters with CUDA-event or CUPTI Activity elapsed time.
4. Join with static SASS/PTX and NVBit features when explaining performance.
5. Treat read-only clock/temperature/throttle data as covariates.

Do not run NCU, lock clocks, change compute mode, or set persistence mode.

## Multiple Contexts and Streams

- Non-default streams are required for concurrent kernel execution.
- Multiple contexts on one GPU usually time-slice; use one context per process
  unless a workload specifically needs otherwise.
- Do not use `nvidia-smi -c EXCLUSIVE_PROCESS` to enforce context policy in
  this project.

## See Also

- `.claude/knowledge/best-practices/perf-coalescing-memory.md`
- `.claude/knowledge/best-practices/perf-instruction-controlflow.md`
- `.claude/knowledge/blackwell/blackwell-tuning.md`
- `.claude/knowledge/cuda/cuda-clusters-dsmem.md`

## Escalate to Raw When

- You need exact per-CC hardware limits.
- You need cluster-aware occupancy edge cases for a new CUDA release.
- A launch failure depends on architecture-conditional Blackwell features.
