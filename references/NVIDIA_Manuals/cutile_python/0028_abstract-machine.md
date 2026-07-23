---
title: "Abstract Machine"
section: ""
source: "https://docs.nvidia.com/cuda/cutile-python/execution.html#abstract-machine"
---

## [Abstract Machine](https://docs.nvidia.com/cuda/cutile-python#abstract-machine)[](https://docs.nvidia.com/cuda/cutile-python/#abstract-machine "Permalink to this headline")

A _tile kernel_ is executed by logical thread [blocks](https://docs.nvidia.com/cuda/cutile-python/#block) organized in a 1D,
2D, or 3D _grid_.

Each _block_ runs on a subset of a GPU defined by the underlying compiler implementation.
Every [block](https://docs.nvidia.com/cuda/cutile-python/#block) executes the body of the [kernel](https://docs.nvidia.com/cuda/cutile-python/#execution-tile-kernels):
scalar operations run serially on a single thread, while array operations
run collectively in parallel across all threads of the [block](https://docs.nvidia.com/cuda/cutile-python/#block).

Tile programs express [block](https://docs.nvidia.com/cuda/cutile-python/#block)-level parallelism only with no exposure to
individual threads within the block.

Explicit synchronization or communication within a [block](https://docs.nvidia.com/cuda/cutile-python/#block) is not permitted,
but is allowed between different [blocks](https://docs.nvidia.com/cuda/cutile-python/#block).

A [block](https://docs.nvidia.com/cuda/cutile-python/#block) defines the unit of execution and a [tile](https://docs.nvidia.com/cuda/cutile-python/data.html#data-tiles-and-scalars) defines unit of data,
which shall not be confused. A single block may operate on multiple [tiles](https://docs.nvidia.com/cuda/cutile-python/data.html#data-tiles-and-scalars)
with different shapes originating from different [global arrays](https://docs.nvidia.com/cuda/cutile-python/data.html#data-global-arrays).
