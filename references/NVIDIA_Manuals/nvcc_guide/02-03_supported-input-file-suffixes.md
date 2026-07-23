---
title: "2.3. Supported Input File Suffixes"
section: "2.3"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#supported-input-file-suffixes"
---

## [2.3. Supported Input File Suffixes](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#supported-input-file-suffixes)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#supported-input-file-suffixes "Permalink to this headline")

The following table defines how `nvcc` interprets its input files:

| Input File Suffix | Description |
| --- | --- |
| `.cu` | CUDA source file, containing host code and device functions |
| `.c` | C source file |
| `.cc`, `.cxx`, `.cpp` | C++ source file |
| `.ptx` | PTX intermediate assembly file (see [Figure 1](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#cuda-compilation-from-cu-to-executable-figure)) |
| `.cubin` | CUDA device code binary file (CUBIN) for a single GPU architecture (see [Figure 1](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#cuda-compilation-from-cu-to-executable-figure)) |
| `.fatbin` | CUDA fat binary file that may contain multiple PTX and CUBIN files (see [Figure 1](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#cuda-compilation-from-cu-to-executable-figure)) |
| `.o`, `.obj` | Object file |
| `.a`, `.lib` | Library file |
| `.res` | Resource file |
| `.so` | Shared object file |

Note that `nvcc` does not make any distinction between object, library or resource files. It just passes files of these types to the linker when the linking phase is executed.
