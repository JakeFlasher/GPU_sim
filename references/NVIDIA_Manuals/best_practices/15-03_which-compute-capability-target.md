---
title: "15.3. Which Compute Capability Target"
section: "15.3"
source: "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#which-compute-capability-target"
---

## [15.3. Which Compute Capability Target](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide#which-compute-capability-target)[](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#which-compute-capability-target "Permalink to this headline")

To target specific versions of NVIDIA hardware and CUDA software, use the `-arch`, `-code`, and `-gencode` options of `nvcc`. Code that uses the warp shuffle operation, for example, must be compiled with `-arch=sm_30` (or higher compute capability).

See [Building for Maximum Compatibility](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#building-for-maximum-compatibility) for further discussion of the flags used for building code for multiple generations of CUDA-capable device simultaneously.
