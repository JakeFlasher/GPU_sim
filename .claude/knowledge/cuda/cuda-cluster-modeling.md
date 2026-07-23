---
Use when: reasoning about thread-block clusters, DSMEM latency, barrier costs, or cluster-related features as inputs to the current non-root measurement and prediction project.
Primary raw source: references/papers/knowledge_base/active/2025__clustersim_modeling_thread_block_clusters_in_hopper_gpus.pdf.
Supporting sources:
- references/NVIDIA_Manuals/cuda_guide/01-02-02-01-01_thread-block-clusters.md
- references/NVIDIA_Manuals/cuda_guide/02-02-03-08_distributed-shared-memory.md
- .claude/knowledge/cuda/cuda-clusters-dsmem.md
Repo touchpoints:
- workloads/simbench_bw/l2/cluster_remote_reduction/
- tools/cloud/preflight.sh
- tools/nvbit_tracer/
Out of scope: NCU cluster metrics, deleted D2 plans, and simulator barrier-state implementation.
Version family: CUDA clusters on CC 9.0+ with SM120 as the primary cloud target.
---

# Cluster Execution for Non-Root Measurement

Thread-block clusters and DSMEM remain relevant because they change wall-clock
behavior and memory-access structure. They are not a reason to revive the old
D2 simulator/modeling plan or to use NCU cluster metrics.

## Current Project Use

Allowed evidence for cluster workloads:

- CUDA launch configuration and cluster dimensions.
- Runtime cluster APIs and launch-probe success/failure.
- CUDA-event or CUPTI Activity elapsed time.
- Static PTX/SASS for `barrier.cluster`, DSMEM, TMA, and related instructions.
- NVBit traces where bounded instrumentation is practical.

Forbidden evidence:

- NCU launch, GPC, or peer-aperture metrics.
- Hardware-counter CUPTI profiler surfaces.
- Deleted `results/cluster_microbench` artifacts as active data.
- Simulator barrier-state validation.

## What to Measure

Useful non-root features include:

- Cluster size and grid divisibility.
- Whether `cudaLaunchKernelEx` accepts the requested cluster shape.
- Dynamic shared-memory size per block and total DSMEM footprint.
- Number and placement of `cluster.sync()` calls.
- DSMEM access pattern: self, pairwise peer, ring, reduction, or contention.
- Static instruction mix for cluster barriers and remote shared-memory access.
- Elapsed-time sensitivity to cluster size and access pattern.

## Prior Art

ClusterSim is retained as modeling prior art. It is useful for deciding which
observables matter: DSMEM latency, barrier overhead, GPC-local scheduling, and
contention. This fork should use those ideas to design non-root measurements
and prediction features, not to recreate a simulator.

## See Also

- `.claude/knowledge/cuda/cuda-clusters-dsmem.md`
- `.claude/knowledge/ptx/ptx-isa-barriers.md`
- `.claude/knowledge/ptx/ptx-isa-async-copy.md`
- `.claude/knowledge/blackwell/rtx_pro_6000_blackwell_capabilities.md`
- `.claude/knowledge/measurement/nonroot-cloud-contract.md`

## Escalate to Raw When

- A CUDA release changes cluster launch attributes or occupancy APIs.
- A paper draft cites specific DSMEM latency or ClusterSim numbers.
- A workload's cluster launch succeeds on one Blackwell SKU but fails on another.
