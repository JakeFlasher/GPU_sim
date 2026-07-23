---
title: "5.7. NVCC Examples"
section: "5.7"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-examples"
---

## [5.7. NVCC Examples](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#nvcc-examples)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-examples "Permalink to this headline")

### [5.7.1. Base Notation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#base-notation)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#base-notation "Permalink to this headline")

`nvcc` provides the options `--gpu-architecture` and `--gpu-code` for specifying the target architectures for both translation stages. Except for allowed short hands described below, the `--gpu-architecture` option takes a single value, which must be the name of a virtual compute architecture, while option `--gpu-code` takes a list of values which must all be the names of actual GPUs. `nvcc` performs a stage 2 translation for each of these GPUs, and will embed the result in the result of compilation (which usually is a host object file or executable).

**Example**

```text
nvcc x.cu --gpu-architecture=compute_80 --gpu-code=sm_80,sm_86
```

### [5.7.2. Shorthand](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#shorthand)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#shorthand "Permalink to this headline")

`nvcc` allows a number of shorthands for simple cases.

#### [5.7.2.1. Shorthand 1](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#shorthand-1)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#shorthand-1 "Permalink to this headline")

`--gpu-code` arguments can be virtual architectures. In this case the stage 2 translation will be omitted for such virtual architecture, and the stage 1 PTX result will be embedded instead. At application launch, and in case the driver does not find a better alternative, the stage 2 compilation will be invoked by the driver with the PTX as input.

**Example**

```text
nvcc x.cu --gpu-architecture=compute_80 --gpu-code=compute_80,sm_80,sm_86
```

#### [5.7.2.2. Shorthand 2](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#shorthand-2)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#shorthand-2 "Permalink to this headline")

The `--gpu-code` option can be omitted. Only in this case, the `--gpu-architecture` value can be a non-virtual architecture. The `--gpu-code` values default to the _closest_ virtual architecture that is implemented by the GPU specified with `--gpu-architecture`, plus the `--gpu-architecture`, value itself. The _closest_ virtual architecture is used as the effective `--gpu-architecture`, value. If the `--gpu-architecture` value is a virtual architecture, it is also used as the effective `--gpu-code` value.

**Example**

```text
nvcc x.cu --gpu-architecture=sm_86
nvcc x.cu --gpu-architecture=compute_80
```

are equivalent to

```text
nvcc x.cu --gpu-architecture=compute_86 --gpu-code=sm_86,compute_86
nvcc x.cu --gpu-architecture=compute_80 --gpu-code=compute_80
```

#### [5.7.2.3. Shorthand 3](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#shorthand-3)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#shorthand-3 "Permalink to this headline")

Both `--gpu-architecture` and `--gpu-code` options can be omitted.

**Example**

```text
nvcc x.cu
```

is equivalent to

```text
nvcc x.cu --gpu-architecture=compute_75 --gpu-code=sm_75,compute_75
```

### [5.7.3. GPU Code Generation in CUDA](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#gpu-code-generation-in-cuda)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-code-generation-in-cuda "Permalink to this headline")

#### [5.7.3.1. List of Supported GPU Codes](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#list-of-supported-gpu-codes)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#list-of-supported-gpu-codes "Permalink to this headline")

Below are the recognized GPU code values for compilation and optimization:

**Compute Capability Targets**

| `compute_75` | `compute_80` | `compute_86` | `compute_87` |
| --- | --- | --- | --- |
| `compute_88` | `compute_89` | `compute_90` | `compute_90a` |
| `compute_100` | `compute_100f` | `compute_100a` | `compute_103` |
| `compute_103f` | `compute_103a` | `compute_110` | `compute_110f` |
| `compute_110a` | `compute_120` | `compute_120f` | `compute_120a` |
| `compute_121` | `compute_121f` | `compute_121a` |  |

**Link-Time Optimization (LTO) Targets**

| `lto_75` | `lto_80` | `lto_86` | `lto_87` |
| --- | --- | --- | --- |
| `lto_88` | `lto_89` | `lto_90` | `lto_90a` |
| `lto_100` | `lto_100f` | `lto_100a` | `lto_103` |
| `lto_103f` | `lto_103a` | `lto_110` | `lto_110f` |
| `lto_110a` | `lto_120` | `lto_120f` | `lto_120a` |
| `lto_121` | `lto_121f` | `lto_121a` |  |

**Streaming Multiprocessor (SM) Architectures**

| `sm_75` | `sm_80` | `sm_86` | `sm_87` |
| --- | --- | --- | --- |
| `sm_88` | `sm_89` | `sm_90` | `sm_90a` |
| `sm_100` | `sm_100f` | `sm_100a` | `sm_103` |
| `sm_103f` | `sm_103a` | `sm_110` | `sm_110f` |
| `sm_110a` | `sm_120` | `sm_120f` | `sm_120a` |
| `sm_121` | `sm_121f` | `sm_121a` |  |

#### [5.7.3.2. Extended Notation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#id54)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#id54 "Permalink to this headline")

The options `--gpu-architecture` and `--gpu-code` can be used in all cases where code is to be generated for one or more GPUs using a common virtual architecture. This will cause a single invocation of `nvcc` stage 1 (that is, preprocessing and generation of virtual PTX assembly code), followed by a compilation stage 2 (binary code generation) repeated for each specified GPU.

Using a common virtual architecture means that all assumed GPU features are fixed for the entire `nvcc` compilation. For instance, the following `nvcc` command assumes only the features available in compute capability 8.0 for both `sm_80` code and the `sm_86` code:

```text
nvcc x.cu --gpu-architecture=compute_80 --gpu-code=compute_80,sm_80,sm_86
```

Sometimes it is necessary to perform different GPU code generation steps, partitioned over different architectures. This is possible using `nvcc` option `--generate-code`, which then must be used instead of a `--gpu-architecture` and `--gpu-code` combination.

#### [5.7.3.3. Using Code Generation Options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#using-code-generation-options)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#using-code-generation-options "Permalink to this headline")

Unlike option `--gpu-architecture`, option `--generate-code` may be repeated on the `nvcc` command line. It takes sub-options `arch` and `code`, which must not be confused with their main option equivalents, but behave similarly. If repeated architecture compilation is used, then the device code must use conditional compilation based on the value of the architecture identification macro `__CUDA_ARCH__`, which is described in the next section.

For example, the following assumes only the features available in compute capability 8.0 for the `sm_80` and `sm_86` code, but all features available in compute capability 9.0 support on `sm_90`:

```text
nvcc x.cu \
    --generate-code arch=compute_80,code=sm_80 \
    --generate-code arch=compute_80,code=sm_86 \
    --generate-code arch=compute_90,code=sm_90
```

Or, leaving actual GPU code generation to the JIT compiler in the CUDA driver:

```text
nvcc x.cu \
    --generate-code arch=compute_80,code=compute_80 \
    --generate-code arch=compute_90,code=compute_90
```

The code sub-options can be combined with a slightly more complex syntax:

```text
nvcc x.cu \
    --generate-code arch=compute_80,code=[sm_80,sm_86] \
    --generate-code arch=compute_90,code=sm_90
```

### [5.7.4. Virtual Architecture Macros](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#virtual-architecture-macros)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#virtual-architecture-macros "Permalink to this headline")

The architecture identification macro `__CUDA_ARCH__` is assigned a three-digit value string `xy0` (ending in a literal `0`) for each stage 1 `nvcc` compilation that compiles for `compute_xy`.

This macro can be used in the implementation of GPU functions for determining the virtual architecture for which it is currently being compiled. The host code (the non-GPU code) must _not_ depend on it.

The architecture list macro `__CUDA_ARCH_LIST__` is a list of comma-separated `__CUDA_ARCH__` values for each of the virtual architectures specified in the compiler invocation. The list is sorted in numerically ascending order.

The macro `__CUDA_ARCH_LIST__` is defined when compiling C, C++ and CUDA source files.

For example, the following nvcc compilation command line will define `__CUDA_ARCH_LIST__` as `800,860,900` :

```text
nvcc x.cu \
--generate-code arch=compute_90,code=sm_90 \
--generate-code arch=compute_80,code=sm_80 \
--generate-code arch=compute_86,code=sm_86 \
--generate-code arch=compute_86,code=sm_89
```
