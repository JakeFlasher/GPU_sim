---
title: "5.6.4.2. Memory in the CUDA Device Runtime"
section: "5.6.4.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#memory-in-the-cuda-device-runtime"
---

### [5.6.4.2. Memory in the CUDA Device Runtime](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#memory-in-the-cuda-device-runtime)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#memory-in-the-cuda-device-runtime "Permalink to this headline")

#### [5.6.4.2.1. Configuration Options](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#configuration-options)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#configuration-options "Permalink to this headline")

Resource allocation for the device runtime system software is controlled via the `cudaDeviceSetLimit()` API from the host program. Limits must be set before any kernel is launched, and may not be changed while the GPU is actively running programs.

The following named limits may be set:

| Limit | Behavior |
| --- | --- |
| `cudaLimitDevRuntimePendingLaunchCount` | Controls the amount of memory set aside for buffering kernel launches and events which have not yet begun to execute, due either to unresolved dependencies or lack of execution resources. When the buffer is full, an attempt to allocate a launch slot during a device side kernel launch will fail and return `cudaErrorLaunchOutOfResources`, while an attempt to allocate an event slot will fail and return `cudaErrorMemoryAllocation`. The default number of launch slots is 2048. Applications may increase the number of launch and/or event slots by setting `cudaLimitDevRuntimePendingLaunchCount`. The number of event slots allocated is twice the value of that limit. |
| `cudaLimitStackSize` | Controls the stack size in bytes of each GPU thread. The CUDA driver automatically increases the per-thread stack size for each kernel launch as needed. This size isn’t reset back to the original value after each launch. To set the per-thread stack size to a different value, `cudaDeviceSetLimit()` can be called to set this limit. The stack will be immediately resized, and if necessary, the device will block until all preceding requested tasks are complete. `cudaDeviceGetLimit()` can be called to get the current per-thread stack size. |

#### [5.6.4.2.2. Allocation and Lifetime](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#allocation-and-lifetime)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#allocation-and-lifetime "Permalink to this headline")

`cudaMalloc()` and `cudaFree()` have distinct semantics between the host and device environments. When invoked from the host, `cudaMalloc()` allocates a new region from unused device memory. When invoked from the device runtime these functions map to device-side `malloc()` and `free()`. This implies that within the device environment the total allocatable memory is limited to the device `malloc()` heap size, which may be smaller than the available unused device memory. Also, it is an error to invoke `cudaFree()` from the host program on a pointer which was allocated by `cudaMalloc()` on the device or vice-versa.

|  | `cudaMalloc()` on Host | `cudaMalloc()` on Device |
| --- | --- | --- |
| `cudaFree()` on Host | Supported | Not Supported |
| `cudaFree()` on Device | Not Supported | Supported |
| Allocation limit | Available device memory | `cudaLimitMallocHeapSize` |

##### [5.6.4.2.2.1. Memory Declarations](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#memory-declarations)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#memory-declarations "Permalink to this headline")

###### [5.6.4.2.2.1.1. Device and Constant Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#device-and-constant-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#device-and-constant-memory "Permalink to this headline")

Memory declared at file scope with `__device__` or `__constant__` memory space specifiers behaves identically when using the device runtime. All kernels may read or write device variables, whether the kernel was initially launched by the host or device runtime. Equivalently, all kernels will have the same view of `__constant__`s as declared at the module scope.

###### [5.6.4.2.2.1.2. Textures and Surfaces](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#textures-and-surfaces)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#textures-and-surfaces "Permalink to this headline")

> The device runtime does not allow creation or destruction of texture or surface objects from within device code. Texture and surface objects created from the host may be used and passed around freely on the device. Regardless of where they are created, dynamically created texture objects are always valid and may be passed to child kernels from a parent.

> **Note**
>
> The device runtime does not support legacy module-scope (i.e., compute capability 2.0 or Fermi-style) textures and surfaces within a kernel launched from the device. Module-scope (legacy) textures may be created from the host and used in device code as for any kernel, but may only be used by a top-level kernel (i.e., the one which is launched from the host).

###### [5.6.4.2.2.1.3. Shared Memory Variable Declarations](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#shared-memory-variable-declarations)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#shared-memory-variable-declarations "Permalink to this headline")

In CUDA C++ shared memory can be declared either as a statically sized file-scope or function-scoped variable, or as an `extern` variable with the size determined at runtime by the kernel’s caller via a launch configuration argument. Both types of declarations are valid under the device runtime.

```c++
__global__ void permute(int n, int *data) {
   extern __shared__ int smem[];
   if (n <= 1)
       return;

   smem[threadIdx.x] = data[threadIdx.x];
   __syncthreads();

   permute_data(smem, n);
   __syncthreads();

   // Write back to GMEM since we can't pass SMEM to children.
   data[threadIdx.x] = smem[threadIdx.x];
   __syncthreads();

   if (threadIdx.x == 0) {
       permute<<< 1, 256, n/2*sizeof(int) >>>(n/2, data);
       permute<<< 1, 256, n/2*sizeof(int) >>>(n/2, data+n/2);
   }
}

void host_launch(int *data) {
    permute<<< 1, 256, 256*sizeof(int) >>>(256, data);
}
```

###### [5.6.4.2.2.1.4. Constant Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#constant-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#constant-memory "Permalink to this headline")

Constants may not be modified from the device. They may only be modified from the host, but the behavior of modifying a constant from the host while there is a concurrent grid that access that constant at any point during its lifetime is undefined.

###### [5.6.4.2.2.1.5. Symbol Addresses](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices#symbol-addresses)[](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/#symbol-addresses "Permalink to this headline")

Device-side symbols (i.e., those marked `__device__`) may be referenced from within a kernel simply via the `&` operator, as all global-scope device variables are in the kernel’s visible address space. This also applies to `__constant__` symbols, although in this case the pointer will reference read-only data.

Since device-side symbols can be referenced directly, those CUDA runtime APIs which reference symbols (e.g., `cudaMemcpyToSymbol()` or `cudaGetSymbolAddress()`) are unnecessary and are not supported by the device runtime. This implies that constant data cannot be altered from within a running kernel, even ahead of a child kernel launch, as references to `__constant__` space are read-only.
