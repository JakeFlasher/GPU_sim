---
title: "4.3. NVCC Environment Variables"
section: "4.3"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-environment-variables"
---

## [4.3. NVCC Environment Variables](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#nvcc-environment-variables)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-environment-variables "Permalink to this headline")

**NVCC_PREPEND_FLAGS and NVCC_APPEND_FLAGS:**

The `nvcc` command line flags can be augmented using the following environment variables, if set:

`NVCC_PREPEND_FLAGS`

> Flags to be injected before the normal nvcc command line.

`NVCC_APPEND_FLAGS`

> Flags to be injected after the normal nvcc command line.

For example, after setting:

```text
export NVCC_PREPEND_FLAGS='-G -keep -arch=sm_90'

export NVCC_APPEND_FLAGS='-DNAME=" foo "'
```

The following invocation:

```text
nvcc foo.cu -o foo
```

Becomes equivalent to:

```text
nvcc -G -keep -arch=sm_90 foo.cu -o foo -DNAME=" foo "
```

These environment variables can be useful for injecting `nvcc` flags globally without modifying build scripts.

The additional flags coming from either NVCC_PREPEND_FLAGS or NVCC_APPEND_FLAGS will be listed in the verbose log (`--verbose`).

**NVCC_CCBIN:**

A default host compiler can be set using the environment variable `NVCC_CCBIN`. For example, after setting:

```text
export NVCC_CCBIN='gcc'
```

`nvcc` will choose `gcc` as the host compiler if `--compiler-bindir` is not set.

`NVCC_CCBIN` can be useful for controlling the default host compiler globally. If `NVCC_CCBIN` and `--compiler-bindir` are both set, `nvcc` will choose the host compiler specified by `--compiler-bindir`. For example:

```text
export NVCC_CCBIN='gcc'

nvcc foo.cu -ccbin='clang' -o foo
```

In this case, `nvcc` will choose `clang` as the host compiler.
