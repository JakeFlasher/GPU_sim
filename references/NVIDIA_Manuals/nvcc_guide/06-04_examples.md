---
title: "6.4. Examples"
section: "6.4"
source: "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#examples"
---

## [6.4. Examples](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc#examples)[](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#examples "Permalink to this headline")

Suppose we have the following files:

```c++
//---------- b.h ----------
#define N 8

extern __device__ int g[N];

extern __device__ void bar(void);
```

```c++
//---------- b.cu ----------
#include "b.h"

__device__ int g[N];

__device__ void bar (void)
{
  g[threadIdx.x]++;
}
```

```c++
//---------- a.cu ----------
#include <stdio.h>
#include "b.h"

__global__ void foo (void) {

  __shared__ int a[N];
  a[threadIdx.x] = threadIdx.x;

  __syncthreads();

  g[threadIdx.x] = a[blockDim.x - threadIdx.x - 1];

  bar();
}

int main (void) {
  unsigned int i;
  int *dg, hg[N];
  int sum = 0;

  foo<<<1, N>>>();

  if(cudaGetSymbolAddress((void**)&dg, g)){
      printf("couldn't get the symbol addr\n");
      return 1;
  }
  if(cudaMemcpy(hg, dg, N * sizeof(int), cudaMemcpyDeviceToHost)){
      printf("couldn't memcpy\n");
      return 1;
  }

  for (i = 0; i < N; i++) {
    sum += hg[i];
  }
  if (sum == 36) {
    printf("PASSED\n");
  } else {
    printf("FAILED (%d)\n", sum);
  }

  return 0;
}
```

These can be compiled with the following commands (these examples are for Linux):

```text
nvcc --gpu-architecture=sm_100 --device-c a.cu b.cu
nvcc --gpu-architecture=sm_100 a.o b.o
```

If you want to invoke the device and host linker separately, you can do:

```text
nvcc --gpu-architecture=sm_100 --device-c a.cu b.cu
nvcc --gpu-architecture=sm_100 --device-link a.o b.o --output-file link.o
g++ a.o b.o link.o --library-path=<path> --library=cudart
```

Note that all desired target architectures must be passed to the device linker, as that specifies what will be in the final executable (some objects or libraries may contain device code for multiple architectures, and the link step can then choose what to put in the final executable).

If you want to use the driver API to load a linked cubin, you can request just the cubin:

```text
nvcc --gpu-architecture=sm_100 --device-link a.o b.o \
    --cubin --output-file link.cubin
```

The objects could be put into a library and used with:

```text
nvcc --gpu-architecture=sm_100 --device-c a.cu b.cu
nvcc --lib a.o b.o --output-file test.a
nvcc --gpu-architecture=sm_100 test.a
```

Note that only static libraries are supported by the device linker.

A PTX file can be compiled to a host object file and then linked by using:

```text
nvcc --gpu-architecture=sm_100 --device-c a.ptx
```

An example that uses libraries, host linker, and dynamic parallelism would be:

```text
nvcc --gpu-architecture=sm_100 --device-c a.cu b.cu
nvcc --gpu-architecture=sm_100 --device-link a.o b.o --output-file link.o
nvcc --lib --output-file libgpu.a a.o b.o link.o
g++ host.o --library=gpu --library-path=<path> \
    --library=cudadevrt --library=cudart
```

It is possible to do multiple device links within a single host executable, as long as each device link is independent of the other. This requirement of independence means that they cannot share code across device executables, nor can they share addresses (e.g., a device function address can be passed from host to device for a callback only if the device link sees both the caller and potential callback callee; you cannot pass an address from one device executable to another, as those are separate address spaces).
