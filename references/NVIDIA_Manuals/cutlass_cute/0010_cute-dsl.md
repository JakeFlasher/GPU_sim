---
title: "CuTe DSL"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/CHANGELOG.html#cute-dsl"
---

### [CuTe DSL](https://docs.nvidia.com/cutlass/latest#cute-dsl)[](https://docs.nvidia.com/cutlass/latest/#cute-dsl "Permalink to this headline")

- New features
  - Auto-deduced smem size for launching kernels
  - Launch config `smem` now defaults to `None` for auto-calculating kernel shared memory usage, which is recommended unless manual control is required.
  - Warnings will be raised when the manually set shared memory size is insufficient or exceeds the GPU maximum.
  - The default shared memory usage calculation aligns with CUDA C++ static shared memory behavior, i.e. summing all allocations additively.
  - An additional launch option `smem_merge_branch_allocs` is provided to merge shared memory allocations across mutually exclusive code branches, which is recommended for inlined mega-kernels to reduce total footprint.
- Bug fixing and improvements
  - Improved source code correlation for profiling/debugging
