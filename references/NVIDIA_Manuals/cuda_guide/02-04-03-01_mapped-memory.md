---
title: "2.4.3.1. Mapped Memory"
section: "2.4.3.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#mapped-memory"
---

### [2.4.3.1. Mapped Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#mapped-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#mapped-memory "Permalink to this headline")

On systems with [HMM](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-heterogeneous-memory-management) or [ATS](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-unified-address-translation-services), all host memory is directly accessible from the GPU using the host pointers.  When ATS or HMM are not available, host allocations can be made accessible to the GPU by _mapping_ the memory into the GPU’s memory space. Mapped memory is always page-locked.

The code examples which follow will illustrate the following array copy kernel operating directly on mapped host memory.

```cuda
__global__ void copyKernel(float* a, float* b)
{
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        a[idx] = b[idx];
}
```

While mapped memory may be useful in some cases where certain data which is not copied to the GPU needs to be accessed from a kernel, accessing mapped memory in a kernel requires transactions across the CPU-GPU interconnect, PCIe, or NVLink C2C. These operations have higher latency and lower bandwidth compared to accessing device memory. Mapped memory should not be considered a performant alternative to [unified memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-unified-memory) or [explicit memory management](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#intro-cpp-explicit-memory-management) for the majority of a kernel’s memory needs.

#### [2.4.3.1.1. cudaMallocHost and cudaHostAlloc](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#cudamallochost-and-cudahostalloc)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#cudamallochost-and-cudahostalloc "Permalink to this headline")

Host memory allocated with `cudaHostMalloc` or `cudaHostAlloc` is automatically mapped. The pointers returned by these APIs can be directly used in kernel code to access the memory on the host. The host memory is accessed over the CPU-GPU interconnect.

**cudaMallocHost**

```cuda
void usingMallocHost() {
  float* a = nullptr;
  float* b = nullptr;

  CUDA_CHECK(cudaMallocHost(&a, vLen*sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&b, vLen*sizeof(float)));

  initVector(b, vLen);
  memset(a, 0, vLen*sizeof(float));

  int threads = 256;
  int blocks = vLen/threads;
  copyKernel<<<blocks, threads>>>(a, b);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("Using cudaMallocHost: ");
  checkAnswer(a,b);
}
```

**cudaAllocHost**

```cuda
void usingCudaHostAlloc() {
  float* a = nullptr;
  float* b = nullptr;

  CUDA_CHECK(cudaHostAlloc(&a, vLen*sizeof(float), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc(&b, vLen*sizeof(float), cudaHostAllocMapped));

  initVector(b, vLen);
  memset(a, 0, vLen*sizeof(float));

  int threads = 256;
  int blocks = vLen/threads;
  copyKernel<<<blocks, threads>>>(a, b);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("Using cudaAllocHost: ");
  checkAnswer(a, b);
}
```

#### [2.4.3.1.2. cudaHostRegister](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#cudahostregister)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#cudahostregister "Permalink to this headline")

When ATS and HMM are not available, allocations made by system allocators can still be mapped for access directly from GPU kernels using `cudaHostRegister`. Unlike memory created with CUDA APIs, however, the memory cannot be accessed from the kernel using the host pointer. A pointer in the device’s memory region must be obtained using `cudaHostGetDevicePointer()`, and that pointer must be used for accesses in kernel code.

```cuda
void usingRegister() {
  float* a = nullptr;
  float* b = nullptr;
  float* devA = nullptr;
  float* devB = nullptr;

  a = (float*)malloc(vLen*sizeof(float));
  b = (float*)malloc(vLen*sizeof(float));
  CUDA_CHECK(cudaHostRegister(a, vLen*sizeof(float), 0 ));
  CUDA_CHECK(cudaHostRegister(b, vLen*sizeof(float), 0  ));

  CUDA_CHECK(cudaHostGetDevicePointer((void**)&devA, (void*)a, 0));
  CUDA_CHECK(cudaHostGetDevicePointer((void**)&devB, (void*)b, 0));

  initVector(b, vLen);
  memset(a, 0, vLen*sizeof(float));

  int threads = 256;
  int blocks = vLen/threads;
  copyKernel<<<blocks, threads>>>(devA, devB);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("Using cudaHostRegister: ");
  checkAnswer(a, b);
}
```

#### [2.4.3.1.3. Comparing Unified Memory and Mapped Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics#comparing-unified-memory-and-mapped-memory)[](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#comparing-unified-memory-and-mapped-memory "Permalink to this headline")

Mapped memory makes CPU memory accessible from the GPU, but does not guarantee that all types of access, for example atomics, are supported on all systems. Unified memory guarantees that all access types are supported.

Mapped memory remains in CPU memory, which means all GPU accesses must go through the connection between the CPU and GPU: PCIe or NVLink. Latency of accesses made across these links are significantly higher than access to GPU memory, and total available bandwidth is lower. As such, using mapped memory for all kernel memory accesses is unlikely to fully utilize GPU computing resources.

Unified memory is most often migrated to the physical memory of the processor accessing it. After the first migration, repeated access to the same memory page or cache line by a kernel can utilize the full GPU memory bandwidth.

> **Note**
>
> Mapped memory has also been referred to as _zero-copy_ memory in previous documents.
>
> Prior to all CUDA applications using a [unified virtual address space](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/#memory-unified-virtual-address-space), additional APIs were needed to enable memory mapping (`cudaSetDeviceFlags` with `cudaDeviceMapHost`).  These APIs are no longer needed.
>
> Atomic functions (see [Atomic Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#atomic-functions)) operating on mapped host memory are not atomic from the point of view of the host or other GPUs.
>
> CUDA runtime requires that 1-byte, 2-byte, 4-byte, 8-byte, and 16-byte naturally aligned loads and stores to host memory initiated from the device are preserved as single accesses from the point of view of the host and other devices. On some platforms, atomics to memory may be broken by the hardware into separate load and store operations. These component load and store operations have the same requirements on preservation of naturally aligned accesses. The CUDA runtime does not support a PCI Express bus topology where a PCI Express bridge splits 8-byte naturally aligned operations and NVIDIA is not aware of any topology that splits 16-byte naturally aligned operations.
