---
title: "3. Heterogeneous Computing"
section: "3"
source: "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#heterogeneous-computing"
---

# [3. Heterogeneous Computing](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide#heterogeneous-computing)[](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#heterogeneous-computing "Permalink to this headline")

CUDA programming involves running code on two different platforms concurrently: a _host_ system with one or more CPUs and one or more CUDA-enabled NVIDIA GPU _devices_.

While NVIDIA GPUs are frequently associated with graphics, they are also powerful arithmetic engines capable of running thousands of lightweight threads in parallel. This capability makes them well suited to computations that can leverage parallel execution.

However, the device is based on a distinctly different design from the host system, and it’s important to understand those differences and how they determine the performance of CUDA applications in order to use CUDA effectively.
