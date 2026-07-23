---
title: "4.18.6.1. Kernel Launch APIs"
section: "4.18.6.1"
source: "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html#kernel-launch-apis"
---

### [4.18.6.1. Kernel Launch APIs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#kernel-launch-apis)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#kernel-launch-apis "Permalink to this headline")

Device-side kernel launches can be implemented using the following two APIs accessible from PTX: `cudaLaunchDevice()` and `cudaGetParameterBuffer()`. `cudaLaunchDevice()` launches the specified kernel with the parameter buffer that is obtained by calling `cudaGetParameterBuffer()` and filled with the parameters to the launched kernel. The parameter buffer can be NULL, i.e., no need to invoke `cudaGetParameterBuffer()`, if the launched kernel does not take any parameters.

#### [4.18.6.1.1. cudaLaunchDevice](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#cudalaunchdevice)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cudalaunchdevice "Permalink to this headline")

At the PTX level, `cudaLaunchDevice()`needs to be declared in one of the two forms shown below before it is used.

```c++
// PTX-level Declaration of cudaLaunchDevice() when .address_size is 64
.extern .func(.param .b32 func_retval0) cudaLaunchDevice
(
  .param .b64 func,
  .param .b64 parameterBuffer,
  .param .align 4 .b8 gridDimension[12],
  .param .align 4 .b8 blockDimension[12],
  .param .b32 sharedMemSize,
  .param .b64 stream
)
;
```

The CUDA-level declaration below is mapped to one of the aforementioned PTX-level declarations and is found in the system header file `cuda_device_runtime_api.h`. The function is defined in the `cudadevrt` system library, which must be linked with a program in order to use device-side kernel launch functionality.

```c++
// CUDA-level declaration of cudaLaunchDevice()
extern "C" __device__
cudaError_t cudaLaunchDevice(void *func, void *parameterBuffer,
                             dim3 gridDimension, dim3 blockDimension,
                             unsigned int sharedMemSize,
                             cudaStream_t stream);
```

The first parameter is a pointer to the kernel to be launched, and the second parameter is the parameter buffer that holds the actual parameters to the launched kernel. The layout of the parameter buffer is explained in [Parameter Buffer Layout](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#parameter-buffer-layout), below. Other parameters specify the launch configuration, i.e., as grid dimension, block dimension, shared memory size, and the stream associated with the launch (please refer to [Kernel Configuration](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#execution-configuration) for the detailed description of launch configuration.

#### [4.18.6.1.2. cudaGetParameterBuffer](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics#cudagetparameterbuffer)[](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/#cudagetparameterbuffer "Permalink to this headline")

`cudaGetParameterBuffer()` needs to be declared at the PTX level before it’s used. The PTX-level declaration must be in one of the two forms given below, depending on address size:

```c++
// PTX-level Declaration of cudaGetParameterBuffer() when .address_size is 64
.extern .func(.param .b64 func_retval0) cudaGetParameterBuffer
(
  .param .b64 alignment,
  .param .b64 size
)
;
```

The following CUDA-level declaration of `cudaGetParameterBuffer()` is mapped to the aforementioned PTX-level declaration:

```c++
// CUDA-level Declaration of cudaGetParameterBuffer()
extern "C" __device__
void *cudaGetParameterBuffer(size_t alignment, size_t size);
```

The first parameter specifies the alignment requirement of the parameter buffer and the second parameter the size requirement in bytes. In the current implementation, the parameter buffer returned by `cudaGetParameterBuffer()` is always guaranteed to be 64- byte aligned, and the alignment requirement parameter is ignored. However, it is recommended to pass the correct alignment requirement value - which is the largest alignment of any parameter to be placed in the parameter buffer - to `cudaGetParameterBuffer()` to ensure portability in the future.
