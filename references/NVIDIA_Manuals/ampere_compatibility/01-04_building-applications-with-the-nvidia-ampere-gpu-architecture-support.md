---
title: "1.4. Building Applications with the NVIDIA Ampere GPU Architecture Support"
section: "1.4"
source: "https://docs.nvidia.com/cuda/ampere-compatibility-guide/#building-applications-with-the-nvidia-ampere-gpu-architecture-support"
---

## [1.4. Building Applications with the NVIDIA Ampere GPU Architecture Support](https://docs.nvidia.com/cuda/ampere-compatibility-guide#building-applications-with-the-nvidia-ampere-gpu-architecture-support)[](https://docs.nvidia.com/cuda/ampere-compatibility-guide/#building-applications-with-the-nvidia-ampere-gpu-architecture-support "Permalink to this headline")

Depending on the version of the CUDA Toolkit used for building the application, it can be built to include PTX and/or native cubin for the NVIDIA Ampere GPU architecture. Although it is enough to just include PTX, including native cubin also has the following advantages:

- It saves the end user the time it takes to JIT-compile kernels that are available only as PTX. All kernels which do not have native cubins are JIT-compiled from PTX, including kernels from all the libraries linked to the application, even if those kernels are never launched by the application. Especially when using large libraries, this JIT compilation can take a significant amount of time. The CUDA driver caches the cubins generated as a result of the PTX JIT, so this is mostly a one-time cost for a user, but it is time best avoided whenever possible.
- PTX JIT-compiled kernels often cannot take advantage of architectural features of newer GPUs, meaning that native-compiled cubins may be faster or of greater accuracy.

### [1.4.1. Building Applications Using CUDA Toolkit 10.x or Earlier](https://docs.nvidia.com/cuda/ampere-compatibility-guide#building-applications-using-cuda-toolkit-10-x-or-earlier)[](https://docs.nvidia.com/cuda/ampere-compatibility-guide/#building-applications-using-cuda-toolkit-10-x-or-earlier "Permalink to this headline")

The `nvcc` compiler included with versions 10.x (10.0, 10.1 and 10.2) of the CUDA Toolkit can generate cubins native to the Volta and Turing architectures (compute capability 7.x). When using CUDA Toolkit 10.x, to ensure that `nvcc` will generate cubin files for all recent GPU architectures as well as a PTX version for forward compatibility with future GPU architectures, specify the appropriate `-gencode=` parameters on the `nvcc` command line as shown in the examples below.

**Windows**

```text
nvcc.exe -ccbin "C:\vs2010\VC\bin"
  -Xcompiler "/EHsc /W3 /nologo /O2 /Zi /MT"
  -gencode=arch=compute_52,code=sm_52
  -gencode=arch=compute_60,code=sm_60
  -gencode=arch=compute_61,code=sm_61
  -gencode=arch=compute_70,code=sm_70
  -gencode=arch=compute_75,code=sm_75
  -gencode=arch=compute_75,code=compute_75
  --compile -o "Release\mykernel.cu.obj" "mykernel.cu"
```

**Mac/Linux**

```text
/usr/local/cuda/bin/nvcc
  -gencode=arch=compute_52,code=sm_52
  -gencode=arch=compute_60,code=sm_60
  -gencode=arch=compute_61,code=sm_61
  -gencode=arch=compute_70,code=sm_70
  -gencode=arch=compute_75,code=sm_75
  -gencode=arch=compute_75,code=compute_75
  -O2 -o mykernel.o -c mykernel.cu
```

Alternatively, the simplified `nvcc` command-line option `-arch=sm_XX` can be used. It is a shorthand equivalent to the following more explicit `-gencode=` command-line options used above. `-arch=sm_XX` expands to the following:

```text
-gencode=arch=compute_XX,code=sm_XX
-gencode=arch=compute_XX,code=compute_XX
```

However, while the `-arch=sm_XX` command-line option does result in inclusion of a PTX back-end target binary by default, it can only specify a single target cubin architecture at a time, and it is not possible to use multiple `-arch=` options on the same `nvcc` command line, which is why the examples above use `-gencode=` explicitly.

For CUDA toolkits prior to 10.0, one or more of the `-gencode` options will need to be removed according to the architectures supported by the specific toolkit version (for example, CUDA toolkit 9.x supports architectures up to _60 and _61). The final `-gencode` to generate PTX would also need to be update – for further information and examples see the documentation for the specific CUDA toolkit version.

> **Note**
>
> `compute_XX` refers to a PTX version and `sm_XX` refers to a cubin version. The `arch=` clause of the `-gencode=` command-line option to `nvcc` specifies the front-end compilation target and must always be a PTX version. The `code=` clause specifies the back-end compilation target and can either be cubin or PTX or both. Only the back-end target version(s) specified by the `code=` clause will be retained in the resulting binary; at least one should be PTX to provide compatibility with future architectures.

### [1.4.2. Building Applications Using CUDA Toolkit 11.0](https://docs.nvidia.com/cuda/ampere-compatibility-guide#building-applications-using-cuda-toolkit-11-0)[](https://docs.nvidia.com/cuda/ampere-compatibility-guide/#building-applications-using-cuda-toolkit-11-0 "Permalink to this headline")

With versions 11.0 of the CUDA Toolkit, `nvcc` can generate cubin native to the NVIDIA Ampere GPU architecture (compute capability 8.0). When using CUDA Toolkit 11.0, to ensure that `nvcc` will generate cubin files for all recent GPU architectures as well as a PTX version for forward compatibility with future GPU architectures, specify the appropriate `-gencode=` parameters on the `nvcc` command line as shown in the examples below.

**Windows**

```text
nvcc.exe -ccbin "C:\vs2010\VC\bin"
  -Xcompiler "/EHsc /W3 /nologo /O2 /Zi /MT"
  -gencode=arch=compute_52,code=sm_52
  -gencode=arch=compute_60,code=sm_60
  -gencode=arch=compute_61,code=sm_61
  -gencode=arch=compute_70,code=sm_70
  -gencode=arch=compute_75,code=sm_75
  -gencode=arch=compute_80,code=sm_80
  -gencode=arch=compute_80,code=compute_80
  --compile -o "Release\mykernel.cu.obj" "mykernel.cu"
```

**Mac/Linux**

```text
/usr/local/cuda/bin/nvcc
  -gencode=arch=compute_52,code=sm_52
  -gencode=arch=compute_60,code=sm_60
  -gencode=arch=compute_61,code=sm_61
  -gencode=arch=compute_70,code=sm_70
  -gencode=arch=compute_75,code=sm_75
  -gencode=arch=compute_80,code=sm_80
  -gencode=arch=compute_80,code=compute_80
  -O2 -o mykernel.o -c mykernel.cu
```

> **Note**
>
> `compute_XX` refers to a PTX version and `sm_XX` refers to a cubin version. The `arch=` clause of the `-gencode=` command-line option to `nvcc` specifies the front-end compilation target and must always be a PTX version. The `code=` clause specifies the back-end compilation target and can either be cubin or PTX or both. Only the back-end target version(s) specified by the `code=` clause will be retained in the resulting binary; at least one should be PTX to provide compatibility with future architectures.

### [1.4.3. Independent Thread Scheduling Compatibility](https://docs.nvidia.com/cuda/ampere-compatibility-guide#independent-thread-scheduling-compatibility)[](https://docs.nvidia.com/cuda/ampere-compatibility-guide/#independent-thread-scheduling-compatibility "Permalink to this headline")

NVIDIA GPUs since Volta architecture have _Independent Thread Scheduling_ among threads in a warp. If the developer made assumptions about warp-synchronicity[^2], this feature can alter the set of threads participating in the executed code compared to previous architectures. Please see Compute Capability 7.0 in the Programming Guide for details and corrective actions. To aid migration to the NVIDIA Ampere GPU architecture, developers can opt-in to the Pascal scheduling model with the following combination of compiler options.

```text
nvcc -gencode=arch=compute_60,code=sm_80 ...
```
