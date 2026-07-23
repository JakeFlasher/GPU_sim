---
title: "1.3. Verifying Blackwell Compatibility for Existing Applications"
section: "1.3"
source: "https://docs.nvidia.com/cuda/blackwell-compatibility-guide/#verifying-blackwell-compatibility-for-existing-applications"
---

## [1.3. Verifying Blackwell Compatibility for Existing Applications](https://docs.nvidia.com/cuda/blackwell-compatibility-guide#verifying-blackwell-compatibility-for-existing-applications)[](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/#verifying-blackwell-compatibility-for-existing-applications "Permalink to this headline")

The first step towards making a CUDA application compatible with Blackwell architecture is to check if the application binary already contains compatible GPU code (at least the PTX). The following sections explain how to accomplish this for an already built CUDA application.

### [1.3.1. Applications Built Using CUDA Toolkit 12.8 or Earlier](https://docs.nvidia.com/cuda/blackwell-compatibility-guide#applications-built-using-cuda-toolkit-12-8-or-earlier)[](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/#applications-built-using-cuda-toolkit-12-8-or-earlier "Permalink to this headline")

CUDA applications built using CUDA Toolkit versions 2.1 through 12.8 are compatible with Blackwell GPUs as long as they are built to include PTX versions of their kernels. This can be tested by forcing the PTX to JIT-compile at application load time with following the steps:

- Download and install the latest driver from <https://www.nvidia.com/drivers>.
- Set the environment variable `CUDA_FORCE_PTX_JIT=1`.
- Launch the application.

With `CUDA_FORCE_PTX_JIT=1`, GPU binary code embedded in an application binary is ignored. Instead PTX code for each kernel is JIT-compiled to produce GPU binary code. An application fails to execute if it does not include PTX. This means the application is not Blackwell architecture compatible and needs to be rebuilt for compatibility. On the other hand, if the application works properly with this environment variable set, then the application is Blackwell compatible.

> **Note**
>
> Be sure to unset the `CUDA_FORCE_PTX_JIT` environment variable after testing is done.

### [1.3.2. Applications Built Using CUDA Toolkit 12.8](https://docs.nvidia.com/cuda/blackwell-compatibility-guide#applications-built-using-cuda-toolkit-12-8)[](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/#applications-built-using-cuda-toolkit-12-8 "Permalink to this headline")

CUDA applications built using CUDA Toolkit 12.8 are compatible with Blackwell architecture as long as they are built to include kernels in native cubin (compute capability 10.0) or PTX form or both.
