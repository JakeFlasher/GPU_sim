---
title: "11.5. Concurrent Kernel Execution"
section: "11.5"
source: "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#concurrent-kernel-execution"
---

## [11.5. Concurrent Kernel Execution](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide#concurrent-kernel-execution)[](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#concurrent-kernel-execution "Permalink to this headline")

As described in [Asynchronous and Overlapping Transfers with Computation](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-transfers-and-overlapping-transfers-with-computation), CUDA streams can be used to overlap kernel execution with data transfers. On devices that are capable of concurrent kernel execution, streams can also be used to execute multiple kernels simultaneously to more fully take advantage of the device’s multiprocessors. Whether a device has this capability is indicated by the `concurrentKernels` field of the `cudaDeviceProp` structure (or listed in the output of the `deviceQuery` CUDA Sample). Non-default streams (streams other than stream 0) are required for concurrent execution because kernel calls that use the default stream begin only after all preceding calls on the device (in any stream) have completed, and no operation on the device (in any stream) commences until they are finished.

The following example illustrates the basic technique. Because `kernel1` and `kernel2` are executed in different, non-default streams, a capable device can execute the kernels at the same time.

```c++
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
kernel1<<<grid, block, 0, stream1>>>(data_1);
kernel2<<<grid, block, 0, stream2>>>(data_2);
```
