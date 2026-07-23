---
title: "10.3. Allocation"
section: "10.3"
source: "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#allocation"
---

## [10.3. Allocation](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide#allocation)[](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#allocation "Permalink to this headline")

Device memory allocation and de-allocation via `cudaMalloc()` and `cudaFree()` are expensive operations. It is recommended to use `cudaMallocAsync()` and `cudaFreeAsync()` which are stream ordered pool allocators to manage device memory.
