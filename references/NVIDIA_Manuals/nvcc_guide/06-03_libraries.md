---
title: "6.3. Libraries"
section: "6.3"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#libraries"
---

## [6.3. Libraries](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#libraries)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#libraries "Permalink to this headline")

The device linker has the ability to read the static host library formats (`.a` on Linux and Mac OS X, `.lib` on Windows). It ignores any dynamic (`.so` or `.dll`) libraries. The `--library` and `--library-path` options can be used to pass libraries to both the device and host linker. The library name is specified without the library file extension when the `--library` option is used.

```text
nvcc --gpu-architecture=sm_100 a.o b.o --library-path=<path> --library=foo
```

Alternatively, the library name, including the library file extension, can be used without the `--library` option on Windows.

```text
nvcc --gpu-architecture=sm_100 a.obj b.obj foo.lib --library-path=<path>
```

Note that the device linker ignores any objects that do not have relocatable device code.
