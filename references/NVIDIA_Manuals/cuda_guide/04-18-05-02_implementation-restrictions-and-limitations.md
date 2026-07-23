---
title: "4.18.5.2. Implementation Restrictions and Limitations"
section: "4.18.5.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#implementation-restrictions-and-limitations"
---

### [4.18.5.2. Implementation Restrictions and Limitations](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#implementation-restrictions-and-limitations)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#implementation-restrictions-and-limitations "Permalink to this headline")

Dynamic Parallelism guarantees all semantics described in this document, however, certain hardware and software resources are implementation-dependent and limit the scale, performance and other properties of a program which uses the device runtime.

#### [4.18.5.2.1. Runtime](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#runtime)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#runtime "Permalink to this headline")

##### [4.18.5.2.1.1. Memory Footprint](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#memory-footprint)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#memory-footprint "Permalink to this headline")

The device runtime system software reserves memory for various management purposes, in particular a reservation for tracking pending grid launches. Configuration controls are available to reduce the size of this reservation in exchange for certain launch limitations. See [Configuration Options](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#device-runtime-configuration-options), below, for details.

##### [4.18.5.2.1.2. Pending Kernel Launches](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#pending-kernel-launches)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#pending-kernel-launches "Permalink to this headline")

When a kernel is launched, all associated configuration and parameter data is tracked until the kernel completes. This data is stored within a system-managed launch pool.

The size of the fixed-size launch pool is configurable by calling `cudaDeviceSetLimit()` from the host and specifying `cudaLimitDevRuntimePendingLaunchCount`.
