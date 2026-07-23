---
title: "6. Using Separate Compilation in CUDA"
section: "6"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#using-separate-compilation-in-cuda"
---

# [6. Using Separate Compilation in CUDA](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#using-separate-compilation-in-cuda)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#using-separate-compilation-in-cuda "Permalink to this headline")

Prior to the 5.0 release, CUDA did not support separate compilation, so CUDA code could not call device functions or access variables across files. Such compilation is referred to as _whole program compilation_. We have always supported the separate compilation of host code, it was just the CUDA device code that needed to all be within one file. Starting with CUDA 5.0, separate compilation of device code is supported, but the old whole program mode is still the default, so there are new options to invoke separate compilation.
