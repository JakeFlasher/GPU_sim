---
title: "Memory Model"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/memory_model.html#memory-model"
---

# [Memory Model](https://docs.nvidia.com/cuda/cutile-python#memory-model)[](https://docs.nvidia.com/cuda/cutile-python/#memory-model "Permalink to this headline")

cuTile’s memory model permits the compiler and hardware to reorder operations
for performance. Without explicit synchronization, the ordering of memory
accesses across [blocks](https://docs.nvidia.com/cuda/cutile-python/execution.html#block) is not guaranteed.

To coordinate memory accesses between [blocks](https://docs.nvidia.com/cuda/cutile-python/execution.html#block), cuTile provides two attributes
for atomic operations:

- **Memory Order** — defines the ordering semantics of an atomic operation.
- **Memory Scope** — defines the set of [blocks](https://docs.nvidia.com/cuda/cutile-python/execution.html#block) that participate in ordering.

Synchronization operates at per-element granularity: each element in the array participates
independently in the memory model.

For further details, see the Memory Model section of the [Tile IR documentation](https://docs.nvidia.com/cuda/tile-ir/).
