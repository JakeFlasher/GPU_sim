---
title: "CUTLASS C++"
section: ""
source: "https://docs.nvidia.com/cutlass/latest/CHANGELOG.html#cutlass-c"
---

### [CUTLASS C++](https://docs.nvidia.com/cutlass/latest#cutlass-c)[](https://docs.nvidia.com/cutlass/latest/#cutlass-c "Permalink to this headline")

- Add [example 95](https://github.com/NVIDIA/cutlass/tree/main/examples/95_blackwell_gemm_green_context) to support green context SM partition
  - Enables launching GEMM on stream with partial SM allocation.
- Fix some kernel issues:
  - Fix l2_capacity=0 handling in Blackwell SM100/SM120 kernel templates
  - Fix CUTLASS clang build issues
- Fix some profiler issues:
  - Add missing reference kernels for blockwise GEMM profiler
- Various improvements and fixes from the community and CUTLASS team. Thanks to everyone who submitted PRs!
- Optimal code generation with CUDA toolkit versions 13.2.
