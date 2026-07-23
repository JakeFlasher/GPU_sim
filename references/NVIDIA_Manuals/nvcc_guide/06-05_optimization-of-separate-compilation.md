---
title: "6.5. Optimization Of Separate Compilation"
section: "6.5"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#optimization-of-separate-compilation"
---

## [6.5. Optimization Of Separate Compilation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#optimization-of-separate-compilation)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#optimization-of-separate-compilation "Permalink to this headline")

Separately compiled code may not have as high of performance as whole program code because of the inability to inline code across files. A way to still get optimal performance is to use link-time optimization, which stores intermediate code which is then linked together to perform high level optimizations. This can be done with the `--dlink-time-opt` or `-dlto` option. This option must be specified at both compile and link time. If only some of the files are compiled with `-dlto`, then those will be linked and optimized together while the rest uses the normal separate compilation. A side effect is that this shifts some of the compile time to the link phase, and there may be some scalability issues with really large codes. If you want to compile using `-gencode` to build for multiple arch, use `-dc -gencode arch=compute_NN,code=lto_NN` to specify the intermediate IR to be stored (where `NN` is the SM architecture version). Then use `-dlto` option to link for a specific architecture.

As of CUDA 12.0 there is support for runtime LTO via the `nvJitLink` library.
