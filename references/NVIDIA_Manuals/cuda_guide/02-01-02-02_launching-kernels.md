---
title: "2.1.2.2. Launching Kernels"
section: "2.1.2.2"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#launching-kernels"
---

### [2.1.2.2. Launching Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#launching-kernels)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#launching-kernels "Permalink to this headline")

The number of threads that will execute the kernel in parallel is specified as part of the kernel launch. This is called the execution configuration. Different invocations of the same kernel may use different execution configurations, such as a different number of threads or thread blocks.

There are two ways of launching kernels from CPU code, [triple chevron notation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#intro-cpp-launching-kernels-triple-chevron) and `cudaLaunchKernelEx`. Triple chevron notation, the most common way of launching kernels, is introduced here. An example of launching a kernel using `cudaLaunchKernelEx` is shown and discussed in detail in in section [Section 3.1.1](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html#advanced-host-cudalaunchkernelex).

#### [2.1.2.2.1. Triple Chevron Notation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#triple-chevron-notation)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#triple-chevron-notation "Permalink to this headline")

Triple chevron notation is a [CUDA C++ Language Extension](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#execution-configuration) which is used to launch kernels. It is called triple chevron because it uses three chevron characters to encapsulate the execution configuration for the kernel launch, i.e. `<<< >>>`. Execution configuration parameters are specified as a comma separated list inside the chevrons, similar to parameters to a function call. The syntax for a kernel launch of the `vecAdd` kernel is shown below.

```cuda
 __global__ void vecAdd(float* A, float* B, float* C)
 {

 }

int main()
{
    ...
    // Kernel invocation
    vecAdd<<<1, 256>>>(A, B, C);
    ...
}
```

The first two parameters to the triple chevron notation are the grid dimensions and the thread block dimensions, respectively. When using 1-dimensional thread blocks or grids, integers can be used to specify dimensions.

The above code launches a single thread block containing 256 threads. Each thread will execute the exact same kernel code. In [Thread and Grid Index Intrinsics](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#intro-cpp-thread-indexing), we’ll show how each thread can use its index within the thread block and grid to change the data it operates on.

There is a limit to the number of threads per block, since all threads of a block reside on the same streaming multiprocessor(SM) and must share the resources of the SM. On current GPUs, a thread block may contain up to 1024 threads. If resources allow, more than one thread block can be scheduled on an SM simultaneously.

Kernel launches are asynchronous with respect to the host thread. That is, the kernel will be setup for execution on the GPU, but the host code will not wait for the kernel to complete (or even start) executing on the GPU before proceeding. Some form of synchronization between the GPU and CPU must be used to determine that the kernel has completed. The most basic version, completely synchronizing the entire GPU, is shown in [Synchronizing CPU and GPU](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#intro-synchronizing-the-gpu). More sophisticated methods of synchronization are covered in [Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#asynchronous-execution).

When using 2 or 3-dimensional grids or thread blocks, the CUDA type `dim3` is used as the grid and thread block dimension parameters. The code fragment below shows a kernel launch of a `MatAdd` kernel using 16 by 16 grid of thread blocks, each thread block is 8 by 8.

```cuda
int main()
{
    ...
    dim3 grid(16,16);
    dim3 block(8,8);
    MatAdd<<<grid, block>>>(A, B, C);
    ...
}
```
