---
title: "1.2. CUDA Best Practices"
section: "1.2"
source: "https://docs.nvidia.com/cuda/ampere-tuning-guide/#cuda-best-practices"
---

## [1.2. CUDA Best Practices](https://docs.nvidia.com/cuda/ampere-tuning-guide#cuda-best-practices)[](https://docs.nvidia.com/cuda/ampere-tuning-guide/#cuda-best-practices "Permalink to this headline")

The performance guidelines and best practices described in the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) and the [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) apply to all CUDA-capable GPU architectures. Programmers must primarily focus on following those recommendations to achieve the best performance.

The high-priority recommendations from those guides are as follows:

- Find ways to parallelize sequential code.
- Minimize data transfers between the host and the device.
- Adjust kernel launch configuration to maximize device utilization.
- Ensure global memory accesses are coalesced.
- Minimize redundant accesses to global memory whenever possible.
- Avoid long sequences of diverged execution by threads within the same warp.
