---
title: "5.1. GPU Generations"
section: "5.1"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-generations"
---

## [5.1. GPU Generations](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#gpu-generations)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-generations "Permalink to this headline")

In order to allow for architectural evolution, NVIDIA GPUs are released in different generations. New generations introduce major improvements in functionality and/or chip architecture, while GPU models within the same generation show minor configuration differences that _moderately_ affect functionality, performance, or both.

All NVIDIA GPUs have a compute capability (CC) which is a two part identifier in the form `major.minor` The major version identifies the GPU generation, while the minor number identifies the version within that generation. For example, compute capability 8.6 is part of CC 8.x generation. While GPUs generations are often refered to by product names such as Hopper or Blackwell, these don’t always have a direct correspondence to major compute capability version.

Binary compatibility of GPU applications is not guaranteed across different major compute capabilities. For example, a CUDA application that has been compiled for a CC 9.0 GPU will not run on a CC 10.0 GPU (and vice versa). This is because the instruction set and instruction encodings are different between different major compute capabilities.

Binary compatibility within a major compute capability can be guaranteed under certain conditions. This is the case when two GPU versions do not show functional differences (for instance when one version is a scaled down version of the other), or when one version is functionally included in the other. An example of the latter is that code compiled for the target `sm_80` will run on all other CC 8.x GPUs, such as `sm_86` or `sm_89`.
