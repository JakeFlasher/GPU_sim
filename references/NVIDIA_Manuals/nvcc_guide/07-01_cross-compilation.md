---
title: "7.1. Cross Compilation"
section: "7.1"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#cross-compilation"
---

## [7.1. Cross Compilation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#cross-compilation)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#cross-compilation "Permalink to this headline")

Cross compilation is controlled by using the following `nvcc` command line option:

- `--compiler-bindir` is used for cross compilation, where the underlying host compiler is capable of generating objects for the target platform.

On an x86 system, if a CUDA toolkit installation has been configured to support cross compilation to both Tegra and non-Tegra ARM targets, then `nvcc` will use the non-Tegra configuration by default, when an ARM host cross compiler has been specified. To use the Tegra configuration instead, pass “`-target-dir aarch64-linux`” to nvcc.
