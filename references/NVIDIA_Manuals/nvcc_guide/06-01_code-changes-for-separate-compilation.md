---
title: "6.1. Code Changes for Separate Compilation"
section: "6.1"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#code-changes-for-separate-compilation"
---

## [6.1. Code Changes for Separate Compilation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#code-changes-for-separate-compilation)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#code-changes-for-separate-compilation "Permalink to this headline")

The code changes required for separate compilation of device code are the same as what you already do for host code, namely using `extern` and `static` to control the visibility of symbols. Note that previously `extern` was ignored in CUDA code; now it will be honored. With the use of `static` it is possible to have multiple device symbols with the same name in different files. For this reason, the CUDA API calls that referred to symbols by their string name are deprecated; instead the symbol should be referenced by its address.
