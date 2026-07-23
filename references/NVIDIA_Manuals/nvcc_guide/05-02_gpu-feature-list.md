---
title: "5.2. GPU Feature List"
section: "5.2"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-feature-list"
---

## [5.2. GPU Feature List](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#gpu-feature-list)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-feature-list "Permalink to this headline")

The following table lists the names of the current GPU architectures, annotated with the functional capabilities that they provide. There are other differences, such as amounts of register and processor clusters, that only affect execution performance.

In NVCC, GPUs are named `sm_xy`, where `x` denotes the major compute capability, and `y` the minor compute capability. Additionally, to facilitate comparing GPU capabilities, CUDA attempts to choose its GPU names such that if `x1y1` <= `x2y2` then all non-ISA related capabilities of `sm_x1y1` are included in `sm_x2y2`.

| `sm_75` | Turing support |
| --- | --- |
| `sm_80`, `sm_86` and `sm_87`, `sm_88` | NVIDIA Ampere GPU architecture support |
| `sm_89` | Ada support |
| `sm_90`, `sm_90a` | Hopper support |
| `sm_100`, `sm_100f`, `sm_100a`, `sm_103`, `sm_103f`, `sm_103a`, `sm_110`, `sm_110f`, `sm_110a`, `sm_120`, `sm_120f`, `sm_120a`, `sm_121`, `sm_121f`, `sm_121a` | Blackwell support |
