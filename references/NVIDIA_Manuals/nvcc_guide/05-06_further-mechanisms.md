---
title: "5.6. Further Mechanisms"
section: "5.6"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#further-mechanisms"
---

## [5.6. Further Mechanisms](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#further-mechanisms)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#further-mechanisms "Permalink to this headline")

Clearly, compilation staging in itself does not help towards the goal of application compatibility with future GPUs. For this we need two other mechanisms: just-in-time compilation (JIT) and fatbinaries.

### [5.6.1. Just-in-Time Compilation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#just-in-time-compilation)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#just-in-time-compilation "Permalink to this headline")

The compilation step to an actual GPU binds the code to one generation of GPUs. Within that generation, it involves a choice between GPU _coverage_ and possible performance. For example, compiling to `sm_80` allows the code to run on all NVIDIA Ampere and Ada generation GPUs, but compiling to `sm_89` would probably yield better code if Ada generation GPUs are the only targets.

![Just-in-Time Compilation of Device Code](images/further-mechanisms_1.png)

Just-in-Time Compilation of Device Code

By specifying a virtual code architecture instead of a _real_ GPU, `nvcc` postpones the assembly of PTX code until application runtime, at which time the target GPU is exactly known. For instance, the command below allows generation of exactly matching GPU binary code, when the application is launched on an `sm_90` or later architecture.

```text
nvcc x.cu --gpu-architecture=compute_90 --gpu-code=compute_90
```

The disadvantage of just-in-time compilation is increased application startup delay, but this can be alleviated by letting the CUDA driver use a compilation cache (refer to “Section 3.1.1.2. Just-in-Time Compilation” of [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)) which is persistent over multiple runs of the applications.

### [5.6.2. Fatbinaries](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#fatbinaries)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#fatbinaries "Permalink to this headline")

A different solution to overcome startup delay by JIT while still allowing execution on newer GPUs is to specify multiple code instances, as in

```text
nvcc x.cu --gpu-architecture=compute_80 --gpu-code=compute_80,sm_86,sm_89
```

This command generates exact code for two architectures, plus PTX code for use by JIT in case a next-generation GPU is encountered. `nvcc` organizes its device code in fatbinaries, which are able to hold multiple translations of the same GPU source code. At runtime, the CUDA driver will select the most appropriate translation when the device function is launched.
