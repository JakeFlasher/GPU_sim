---
title: "1.2. Supported Host Compilers"
section: "1.2"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#supported-host-compilers"
---

## [1.2. Supported Host Compilers](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#supported-host-compilers)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#supported-host-compilers "Permalink to this headline")

A general purpose C++ host compiler is needed by `nvcc` in the following situations:

- During non-CUDA phases (except the run phase), because these phases will be forwarded by `nvcc` to this compiler.
- During CUDA phases, for several preprocessing stages and host code compilation (see also [The CUDA Compilation Trajectory](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#cuda-compilation-trajectory)).

`nvcc` assumes that the host compiler is installed with the standard method designed by the compiler provider. If the host compiler installation is non-standard, the user must make sure that the environment is set appropriately and use relevant `nvcc` compile options.

The following documents provide detailed information about supported host compilers:

- [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [NVIDIA CUDA Installation Guide for Microsoft Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

On all platforms, the default host compiler executable (`gcc` and `g++` on Linux and `cl.exe` on Windows) found in the current execution search path will be used, unless specified otherwise with appropriate options (see *file-and-path-specifications*).

Note, `nvcc` does not support the compilation of file paths that exceed the maximum path length limitations of the host system. To support the compilation of long file paths, please refer to the documentation for your system.
