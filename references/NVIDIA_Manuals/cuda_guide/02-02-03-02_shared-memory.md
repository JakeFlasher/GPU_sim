---
title: "2.2.3.2. Shared Memory"
section: "2.2.3.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory"
---

### [2.2.3.2. Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#shared-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#shared-memory "Permalink to this headline")

Shared memory is a memory space that is accessible by all threads in a thread block. It is physically located on each SM and uses the same physical resource as the L1 cache, the unified data cache.  The data in shared memory persists throughout the kernel execution.  Shared memory can be considered a user-managed scratchpad for use during kernel execution.  While small in size compared to global memory, because shared memory is located on each SM, the bandwidth is higher and the latency is lower than accessing global memory.

Since shared memory is accessible by all threads in a thread block, care must be taken to avoid data races between threads in the same thread block.  Synchronization between threads in the same thread block can be achieved using the `__syncthreads()` function.  This function blocks all threads in the thread block until all threads have reached the call to `__syncthreads()`.

```cuda
// assuming blockDim.x is 128
__global__ void example_syncthreads(int* input_data, int* output_data) {
    __shared__ int shared_data[128];
    // Every thread writes to a distinct element of 'shared_data':
    shared_data[threadIdx.x] = input_data[threadIdx.x];

    // All threads synchronize, guaranteeing all writes to 'shared_data' are ordered
    // before any thread is unblocked from '__syncthreads()':
    __syncthreads();

    // A single thread safely reads 'shared_data':
    if (threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            sum += shared_data[i];
        }
        output_data[blockIdx.x] = sum;
    }
}
```

The size of shared memory varies depending on the GPU architecture being used.  Because shared memory and L1 cache share the same physical space, using shared memory reduces the size of the usable L1 cache for a kernel.  Additionally, if no shared memory is used by the kernel, the entire physical space will be utilized by L1 cache.  The CUDA runtime API provides functions to query the shared memory size on a per SM basis and a per thread block basis, using the `cudaGetDeviceProperties` function and investigating the `cudaDeviceProp.sharedMemPerMultiprocessor` and `cudaDeviceProp.sharedMemPerBlock` device properties.

The CUDA runtime API provides a function `cudaFuncSetCacheConfig` to tell the runtime whether to allocate more space to shared memory, or more space to L1 cache.  This function specifies a preference to the runtime, but is not guaranteed to be honored.  The runtime is free to make decisions based on the available resources and the needs of the kernel.

Shared memory can be allocated both statically and dynamically.

#### [2.2.3.2.1. Static Allocation of Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#static-allocation-of-shared-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#static-allocation-of-shared-memory "Permalink to this headline")

To allocate shared memory statically, the programmer must declare a variable inside the kernel using the `__shared__` specifier.  The variable will be allocated in shared memory and will persist for the duration of the kernel execution.  The size of the shared memory declared in this way must be specified at compile time.  For example, the following code snippet, located in the body of the kernel, declares a shared memory array of type `float` with 1024 elements.

```c++
__shared__ float sharedArray[1024];
```

After this declaration, all the threads in the thread block will have access to this shared memory array.  Care must be taken to avoid data races between threads in the same thread block, typically with the use of `__syncthreads()`.

#### [2.2.3.2.2. Dynamic Allocation of Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#dynamic-allocation-of-shared-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#dynamic-allocation-of-shared-memory "Permalink to this headline")

To allocate shared memory dynamically, the programmer can specify the desired amount of shared memory per thread block in bytes as the third (and optional) argument to the kernel launch in the triple chevron notation like this `functionName<<<grid, block, sharedMemoryBytes>>>()`.

Then, inside the kernel, the programmer can use the `extern __shared__` specifier to declare a variable that will be allocated dynamically at kernel launch.

```c++
extern __shared__ float sharedArray[];
```

One caveat is that if one wants multiple dynamically allocated shared memory arrays, the single `extern __shared__` must be partitioned manually using pointer arithmetic.  For example, if one wants the equivalent of the following,

```c++
short array0[128];
float array1[64];
int   array2[256];
```

in dynamically allocated shared memory, one could declare and initialize the arrays in the following way:

```c++
extern __shared__ float array[];

short* array0 = (short*)array;
float* array1 = (float*)&array0[128];
int*   array2 =   (int*)&array1[64];
```

Note that pointers need to be aligned to the type they point to, so the following code, for example, does not work since `array1` is not aligned to 4 bytes.

```c++
extern __shared__ float array[];
short* array0 = (short*)array;
float* array1 = (float*)&array0[127];
```
