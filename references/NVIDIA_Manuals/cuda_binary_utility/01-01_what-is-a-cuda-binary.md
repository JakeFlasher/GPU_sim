---
title: "1.1. What is a CUDA Binary?"
section: "1.1"
source: "https://docs.nvidia.com/cuda/cuda-binary-utilities/#what-is-a-cuda-binary"
---

## [1.1. What is a CUDA Binary?](https://docs.nvidia.com/cuda/cuda-binary-utilities#what-is-a-cuda-binary)[](https://docs.nvidia.com/cuda/cuda-binary-utilities/#what-is-a-cuda-binary "Permalink to this headline")

A CUDA binary (also referred to as cubin) file is an ELF-formatted file which consists of CUDA executable code sections as well as other sections containing symbols, relocators, debug info, etc. By default, the CUDA compiler driver `nvcc` embeds cubin files into the host executable file. But they can also be generated separately by using the “`-cubin`” option of `nvcc`. cubin files are loaded at run time by the CUDA driver API.

> **Note**
>
> For more details on cubin files or the CUDA compilation trajectory, refer to [NVIDIA CUDA Compiler Driver NVCC](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).
