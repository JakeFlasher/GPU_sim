---
name: optimizing-cuda-kernels
description: "Diagnose CUDA performance using coalescing, occupancy, dependencies, memory traffic and synchronization. Use for a concrete kernel or architecture experiment, not automatic simulator calibration."
---

# CUDA kernel optimization

Start from one workload/shape, its output correctness, and the leading
bottleneck hypothesis. Inspect the generated code and run a bounded probe
before broad parameter sweeps. Keep a negative result when an optimization
fails. Native GPU execution and NCU are ordinary tools, subject to root policy.

Use `references/NVIDIA_Manuals/best_practices/` and `cuda_guide/` for memory,
occupancy, register allocation, launch and synchronization contracts. Use
`ampere_tuning/` for SM86-local facts. Optional `.claude/knowledge/` cards are
not required or currently present.

Pin workload/tool/device details for measurements. A preferred carveout is not
the driver-selected configuration, occupancy is not throughput, and a
coalescing/topology count is not a completed cache/protocol transaction count.
When optimizing the simulator rather than a kernel, preserve event/frontier
semantics and never fit the scored NCU workload into its model inputs.
