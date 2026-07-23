---
title: "7.1. Build Instruction"
section: "7.1"
source: "https://docs.nvidia.com/cuda/ptx-compiler-api/#build-instruction"
---

## [7.1. Build Instruction](https://docs.nvidia.com/cuda/ptx-compiler-api#build-instruction)[](https://docs.nvidia.com/cuda/ptx-compiler-api/#build-instruction "Permalink to this headline")

Assuming the environment variable `CUDA_PATH` points to CUDA Toolkit installation directory, build this example as:

- Windows:

```text
cl.exe simpleVectorAddition.c /FesimpleVectorAddition ^
          /I "%CUDA_PATH%"\include ^
          "%CUDA_PATH%"\lib\x64\nvptxcompiler_static.lib
          "%CUDA_PATH%"\lib\x64\cuda.lib
```

OR

```text
nvcc simpleVectorAddition.c  -ccbin <PATH_TO_cl.exe>
          -I $CUDA_PATH/include -L $CUDA_PATH/lib/x64/ -lcuda  nvptxcompiler_static.lib
```
- Linux:

```text
gcc simpleVectorAddition.c -o simpleVectorAddition \
             -I $CUDA_PATH/include \
             -L $CUDA_PATH/lib64 \
             libnvptxcompiler_static.a -lcuda -lm -lpthread \
             -Wl,-rpath,$CUDA_PATH/lib64
```
